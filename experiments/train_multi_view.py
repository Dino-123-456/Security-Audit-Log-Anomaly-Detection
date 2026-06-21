# experiments/train_multi_view.py
import os
import sys
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from data.data_loader import BGLDataLoader
from models.multi_view_detector import MultiViewDetector
from utils.logger import get_logger 

def evaluate_multi_view(model, test_loader, train_loader, device, val_loader=None):
    """
    工业级多视角评估 (Multi-View Evaluation):
    1. 语义视角 (Semantic): 基于特征 KNN 的局部密度 (捕捉上下文异常)
    2. 统计视角 (Statistical): 基于训练集 Token 稀有度 (OOV/Rare token penalty，对抗概念漂移)
    3. 融合打分 (Ensemble): 加权融合，并计算工业级指标 Precision@K
    """
    model.eval()
    
    # ================= 1. 构建参考数据库 (特征 & Token 频率) =================
    print("Building reference database (Features & Token Frequencies)...")
    train_features = []
    train_token_counts = Counter()
    
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader, desc="Processing Train Set"):
            inputs = inputs.to(device)
            z = model(inputs)
            train_features.append(z.cpu())
            # 统计训练集 Token 频率 (排除 PAD=0)
            valid_tokens = inputs[inputs != 0].cpu().numpy()
            train_token_counts.update(valid_tokens)
            
    train_features = torch.cat(train_features, dim=0)
    total_train_tokens = sum(train_token_counts.values())
    
    # ================= 2. 提取测试集特征 & 计算统计视角分数 =================
    all_features = []
    all_labels = []
    all_rare_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Extracting Test Features"):
            inputs = inputs.to(device)
            z = model(inputs)
            all_features.append(z.cpu())
            all_labels.extend(labels.numpy())
            
            # 计算统计视角分数 (Rare Token Penalty)
            batch_rare_scores = []
            for seq in inputs.cpu().numpy():
                valid_seq = seq[seq != 0]
                if len(valid_seq) == 0:
                    batch_rare_scores.append(0.0)
                    continue
                
                rare_penalty = 0.0
                for token in valid_seq:
                    freq = train_token_counts.get(token, 0)
                    # 核心逻辑：如果 token 在训练集没见过 (OOV)，或者频率极低，给予高分惩罚
                    if freq == 0:
                        rare_penalty += 10.0  # OOV 强惩罚
                    else:
                        rare_penalty += 1.0 / (freq / total_train_tokens + 1e-6)
                batch_rare_scores.append(rare_penalty / len(valid_seq))
            all_rare_scores.extend(batch_rare_scores)
            
    test_features = torch.cat(all_features, dim=0)
    labels = np.array(all_labels)
    rare_scores = np.array(all_rare_scores)
    
    # ================= 3. 计算语义视角分数 (KNN Local Density) =================
    print("Computing Semantic Scores (KNN)...")
    K = 10
    # 为了计算效率，随机采样部分训练集作为参考库
    ref_size = min(5000, train_features.size(0))
    ref_indices = torch.randperm(train_features.size(0))[:ref_size]
    ref_features = train_features[ref_indices].to(device)
    
    semantic_scores = []
    for i in range(0, test_features.size(0), 512):
        batch_feat = test_features[i:i+512].to(device)
        # 计算余弦相似度 (因为特征已 L2 归一化，点积即余弦相似度)
        sim_matrix = torch.mm(batch_feat, ref_features.t()) 
        topk_sim, _ = torch.topk(sim_matrix, k=min(K, ref_size), dim=1)
        avg_topk_sim = topk_sim.mean(dim=1).cpu().numpy()
        semantic_scores.extend(1.0 - avg_topk_sim) # 相似度越低，异常分越高
        
    semantic_scores = np.array(semantic_scores)
    
    # ================= 4. 分数归一化与双视角融合 =================
    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
    sem_norm = min_max_normalize(semantic_scores)
    rare_norm = min_max_normalize(rare_scores)
    
    # 融合策略：在极端概念漂移下，统计视角(OOV)是保底武器，故赋予更高权重
    final_scores = 0.4 * sem_norm + 0.6 * rare_norm 
    
    # ================= 5. 计算传统学术指标 =================
    try:
        auc = roc_auc_score(labels, final_scores) 
    except ValueError:
        auc = 0.5 
        
    normal_scores = final_scores[labels == 0]
    threshold = np.percentile(normal_scores, 99) if len(normal_scores) > 0 else 0.0
        
    preds = (final_scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # ================= 6. 【核心】计算工业级指标：Precision@K =================
    K_values = [100, 500, 1000]
    prec_at_k = {}
    for k in K_values:
        k_actual = min(k, len(final_scores))
        top_k_indices = np.argsort(final_scores)[-k_actual:][::-1] 
        top_k_labels = labels[top_k_indices]
        prec_at_k[f'P@{k}'] = top_k_labels.sum() / k_actual
        
    return {
        'AUC': auc, 'F1': f1, 'Precision': precision, 'Recall': recall,
        'Threshold': threshold, 'Prec_At_K': prec_at_k
    }

def main():
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/experiment_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    eval_cfg = config.get('evaluation', {})
    log_dir = eval_cfg.get('log_dir', 'outputs/logs')
    ckpt_dir = eval_cfg.get('checkpoint_dir', 'outputs/checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 统一命名为 MultiView
    logger = get_logger("MultiView_v12_Hybrid", log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_cfg = config.get('data', {})
    # 兼容旧配置：优先读取 multi_view，如果没有则回退读取 hypersphere
    mv_cfg = config.get('multi_view', config.get('hypersphere', {}))
    
    logger.info("Loading datasets via BGLDataLoader...")
    data_dict = BGLDataLoader(config).load()
    
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    vocab_size = int(data_dict['vocab_size'])
    
    logger.info(f"Using full training set: {len(X_train)} samples.")
    
    batch_size = mv_cfg.get('batch_size', 256) 
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    # drop_last=True 确保 batch size 固定，避免对比学习矩阵维度不一致
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float))
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # 实例化 MultiViewDetector
    model = MultiViewDetector(
        vocab_size=vocab_size, 
        embed_dim=mv_cfg.get('embedding_dim', 64),
        n_heads=mv_cfg.get('n_heads', 4),
        num_layers=mv_cfg.get('num_layers', 2),
        proj_dim=mv_cfg.get('proj_dim', 32),
        max_len=data_cfg.get('window_size', 10)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=mv_cfg.get('learning_rate', 1e-3), weight_decay=mv_cfg.get('weight_decay', 1e-4))
    epochs = mv_cfg.get('epochs', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    patience = mv_cfg.get('early_stopping_patience', 3) 
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    temperature = 0.1 
    
    logger.info(f"Starting Supervised Contrastive Learning (InfoNCE) for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            z = model(inputs) 
            B = z.size(0)
            
            # ================= InfoNCE Loss 核心计算 =================
            # 计算余弦相似度矩阵 (除以 temperature 放大差异)
            sim_matrix = torch.mm(z, z.t()) / temperature 
            
            # 【踩坑修复 1】使用 out-of-place 操作构建 Mask，避免 In-place 破坏计算图
            eye_mask = torch.eye(B, dtype=torch.bool, device=device)
            
            labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = labels_eq.float()
            mask = mask.masked_fill(eye_mask, 0.0) # 排除自身
            
            # 【踩坑修复 2】数值稳定性：减去最大值防止 exp 溢出
            sim_matrix_max, _ = sim_matrix.max(dim=1, keepdim=True)
            sim_matrix_stable = sim_matrix - sim_matrix_max.detach()
            
            exp_sim = torch.exp(sim_matrix_stable)
            exp_sim = exp_sim.masked_fill(eye_mask, 0.0) # 排除自身的 exp
            
            denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
            num_positives = mask.sum(dim=1).clamp(min=1)
            
            # Log-Softmax 计算
            log_prob = sim_matrix_stable - torch.log(denom + 1e-8)
            loss = -(log_prob * mask).sum(dim=1) / num_positives
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step() 
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Contrastive Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early Stopping 逻辑
        if patience > 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict()) 
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"⚠️ Early stopping triggered at Epoch {epoch+1}!")
                    break
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from Epoch with Loss: {best_loss:.6f}")
        
    logger.info("Evaluating on test set (Multi-View: Semantic + Statistical)...")
    results = evaluate_multi_view(model, test_loader, train_loader, device, val_loader=val_loader)
    
    logger.info("="*50)
    logger.info(f"✅ v12 Final Results (Hybrid Multi-View Scoring):")
    logger.info(f"AUC: {results['AUC']:.4f} | Adaptive F1: {results['F1']:.4f}")
    logger.info(f"Precision: {results['Precision']:.4f} | Recall: {results['Recall']:.4f}")
    logger.info("-" * 50)
    logger.info("🎯 Industrial Metrics (Precision@K):")
    for k, p in results['Prec_At_K'].items():
        logger.info(f"   {k}: {p:.4f} (Top {k} alerts accuracy)")
    logger.info("="*50)
    
    # 统一保存路径命名
    save_path = os.path.join(ckpt_dir, "multi_view_model_v12.pth")
    torch.save({
        'model_state_dict': model.state_dict(), 
        'adaptive_thresh': results['Threshold']
    }, save_path)
    logger.info(f"Best model saved to {save_path}")

if __name__ == "__main__":
    main()