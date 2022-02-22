from kwatsch.brain.trainer_ae import AETrainerExtension1Brain


class AETrainerMNIST(AETrainerExtension1Brain):

    def train(self, batch_item, keep_predictions=True, eval_mode=False):
        x = batch_item['image'].to(self.args['device'])
        if not eval_mode:
            self.model.train()
        else:
            self.model.eval()

        self._iters += 1
        z = self.model.encode(x)
        out = self.model.decode(z)
        if self.use_multiple_gpu:
            out = out.to('cuda:1')
            x = x.to('cuda:1')

        loss_ae = self.get_loss(x, out, is_test=False)['loss_ae']
        slice_between = batch_item['slice_between'].to(z.device)
        return_dict_sbi = self.synthesize_batch_images(batch_item=batch_item, z=z, compute_latent_loss=True,
                                                       slice_between=slice_between, is_eval=True)
        _ = self.get_extra_loss(slice_between, return_dict_sbi['s_between_mix'], return_dict_sbi['z_mix'],
                                            z=z, mask=None if not self.args['get_masks'] else batch_item['loss_mask'],
                                            is_test=False)
        self.opt_ae.zero_grad()
        if not eval_mode:
            loss_ae.backward()
            self.opt_ae.step()

        # if lr_scheduler is not None take a scheduling step
        if self.opt_sched_ae is not None:
            self.opt_sched_ae.step()

        self.losses['loss_ae'].append(loss_ae.item())
        self.losses['loss_latent_1'].append(return_dict_sbi['loss_latent'].item())

        slice_inbetween_mix = return_dict_sbi['s_between_mix'].detach().cpu()
        if keep_predictions:
            self.train_predictions = {'z_mix': return_dict_sbi['z_mix'].detach().cpu(),
                                      'pred_alphas': batch_item['alpha_from'].detach(),
                                      'slice_inbetween_mix': slice_inbetween_mix,
                                      'slice_inbetween_05': slice_inbetween_mix,
                                      "reconstruction": out.detach().cpu()}


class AECombinedTrainerMNIST(AETrainerExtension1Brain):

    def get_extra_loss(self, slice_between, s_between_mix, z_mix, z=None, mask=None, is_test=False):
        if self.args['use_loss_annealing']:
            # print("Loss weight {:.6f}".format(self.loss_weights[self.epoch]))
            loss_extra_image = self.loss_weights[self.epoch] * self.get_extra_image_loss(slice_between, s_between_mix,
                                                                                         mask=mask)
        else:
            loss_extra_image = self.args['ex_loss_weight1'] * self.get_extra_image_loss(slice_between, s_between_mix,
                                                                                        mask=mask)
        if self.args['use_extra_latent_loss']:
            loss_extra_latent = 0.5 * self.get_extra_latent_loss(s_between_mix, slice_between, z_mix, is_test=is_test,
                                                                 z=z)
            loss_ae_extra = loss_extra_latent + loss_extra_image
        else:
            loss_ae_extra = loss_extra_image

        if is_test:
            # self.losses_test['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses_test['loss_ae_dist_extra'].append(loss_extra_image.item())
        else:
            # self.losses['loss_ae_extra'].append(loss_ae_extra.item())
            self.losses['loss_ae_dist_extra'].append(loss_extra_image.item())
        return loss_ae_extra