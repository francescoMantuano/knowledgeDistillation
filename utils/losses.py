import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, gamma):
    #alpha, gamma e temperature da modificare per vedere come diversi parametri condizionano i risultati

    ce_loss = F.cross_entropy(student_logits, labels)
    
    #kl_div prende come parametri (log_probs, probs, reduction)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim = 1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean" #aggregazione standard per KD, indica come aggregare la loss, facendo la media per ogni batch
    )

    return gamma * ce_loss + alpha * kd_loss * (temperature ** 2)


def feature_distillation_loss(student_feat, teacher_feat):
    """
    Feature distillation loss based on cosine similarity.
    Assumes features are [B, C, H, W].
    """

    # 1) Global Average Pooling
    student_vec = F.adaptive_avg_pool2d(student_feat, 1).squeeze(-1).squeeze(-1)
    teacher_vec = F.adaptive_avg_pool2d(teacher_feat, 1).squeeze(-1).squeeze(-1)

    # 2) L2 normalization
    student_vec = F.normalize(student_vec, dim=1)
    teacher_vec = F.normalize(teacher_vec, dim=1)

    # 3) Cosine loss
    loss = 1 - F.cosine_similarity(student_vec, teacher_vec, dim=1)

    return loss.mean()