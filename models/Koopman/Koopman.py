import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.Koopman.LossFunction import LossRec, LossPredls, LossPredss

class Koopman(object):
    def __init__(self, args, state_encoder, state_decoder, action_encoder, control_matrix, state_transition_matrix, koopman_buffer):
        self.args = args
        self.phi_x = state_encoder
        self.phi_x_inv = state_decoder
        self.phi_u = action_encoder
        self.B = control_matrix
        self.K = state_transition_matrix
        self.buffer = koopman_buffer

        self.optimizer = torch.optim.Adam(
            list(self.phi_x.parameters()) + list(self.phi_x_inv.parameters()) + list(self.phi_u.parameters()) + list(self.B.parameters()) + list(self.K.parameters()), lr=3e-4
        )

        self.avg_loss_rec_log = []
        self.avg_loss_predls_log = []
        self.avg_loss_predss_log = []
        self.avg_loss_ki_log = []

    def train(self):
        states, next_states, actions = self.buffer.get()
        dataset = TensorDataset(states, next_states, actions)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        for _ in range(self.args.epochs):
            loss_rec_log = []
            loss_predls_log = []
            loss_predss_log = []
            loss_ki_log = []

            for x_now, x_next, u_now in loader:

                self.optimizer.zero_grad()

                #*--- LossRec ---*#
                y_now = self.phi_x(x_now)
                x_now_rec = self.phi_x_inv(y_now)
                loss_rec = LossRec(x_now, x_now_rec)
                #*---------------*#

                #*--- LossPredls ---*#
                y_next = self.phi_x(x_next)
                y_pred_next = self.B(self.phi_u(u_now)) + self.K(y_now)
                loss_predls = LossPredls(y_next, y_pred_next, 10)
                #*------------------*#

                #*--- LossPredss ---*#
                x_pred_next = self.phi_x_inv(y_pred_next)
                loss_predss = LossPredss(x_next, x_pred_next, 10)
                #*------------------*#

                loss_ki = 0.75 * loss_rec + 0.1 * loss_predls + 0.5 * loss_predss
                loss_ki.backward()
                self.optimizer.step()

                loss_rec_log.append(loss_rec.item())
                loss_predls_log.append(loss_predls.item())
                loss_predss_log.append(loss_predss.item())
                loss_ki_log.append(loss_ki.item())

    def save_model(self, path):
        state_encoder_path = os.path.join(path, "state_encoder.pth")
        state_decoder_path = os.path.join(path, "state_decoder.pth")
        action_encoder_path = os.path.join(path, "action_encoder.pth")
        control_matrixr_path = os.path.join(path, "control_matrix.pth")
        state_transition_matrix_path = os.path.join(path, "state_transition_matrix.pth")

        torch.save(self.phi_x.state_dict(), state_encoder_path)
        torch.save(self.phi_x_inv.state_dict(), state_decoder_path)
        torch.save(self.phi_u.state_dict(), action_encoder_path)
        torch.save(self.B.state_dict(), control_matrixr_path)
        torch.save(self.K.state_dict(), state_transition_matrix_path)