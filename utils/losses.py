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
    #normalizzo siccome teacher e student hanno architetture, profondità e scale di attivazione diverse, altrimenti la scala distorgerebbe la loss
    student_feat = F.normalize(student_feat, dim=1)
    teacher_feat = F.normalize(teacher_feat, dim=1)

    #posso anche utilizzare L1 o cosine ma MSE è standard
    loss = F.mse_loss(student_feat, teacher_feat)
    return loss

#potrei anche utilizzare cosine similarity: 
#loss = 1 - (student_feat * teacher_feat).sum(dim=1).mean()
#return loss