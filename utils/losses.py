import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    #alpha e temperature da modificare per vedere come diversi parametri condizionano i risultati

    ce_loss = F.cross_entropy(student_logits, labels)
    
    #kl_div prende come parametri (log_probs, probs, reduction)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim = 1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean" #aggregazione standard per KD, indica come aggregare la loss, facendo la media per ogni batch
    )

    return alpha * ce_loss + (1 - alpha) * kd_loss * (temperature ** 2)


def feature_distillation_loss(student_feat, teacher_feat):
    #posso anche utilizzare L1 o cosine ma MSE è standard
    return F.mse_loss(student_feat, teacher_feat)
