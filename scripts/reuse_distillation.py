"""Full supervised-token loss; retain logits and reuse teacher metrics."""
import torch
import torch.nn.functional as F

class ReusableKLLoss(torch.nn.Module):
    def forward(self, student_input, student_weight, teacher_input, teacher_weight, true_labels):
        student_logits = student_input @ student_weight.t()
        with torch.no_grad():
            teacher_logits = teacher_input @ teacher_weight.t()
            teacher_log_probs = F.log_softmax(teacher_logits,dim=-1)
            self.last_teacher_ce = F.cross_entropy(teacher_logits.float(),true_labels)
        student_log_probs = F.log_softmax(student_logits,dim=-1)
        loss = F.kl_div(student_log_probs,teacher_log_probs,log_target=True,reduction='batchmean')
        with torch.no_grad():
            student_ce = F.nll_loss(student_log_probs.float(),true_labels)
        return loss,loss.detach(),student_ce
