# coding: utf-8
import math
import random
import torch
import numpy as np


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        pose_dim=14,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn
        self.pose, self.pose_lengths = torch_batch.pose

        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            
            tmp_pose = torch.zeros_like(self.pose)
            tmp_pose_lengths = torch.zeros_like(self.pose_lengths)
            
            for idx, (features_sgn, features_pose, length_sgn, length_pose) in enumerate(zip(self.sgn, self.pose, self.sgn_lengths, self.pose_lengths)):
                features_sgn = features_sgn.clone()
                features_pose = features_pose.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data_sgn = features_sgn[: length.long(), :]
                tmp_data_sgn = tmp_data_sgn[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0 : tmp_data_sgn.shape[0]] = tmp_data_sgn
                tmp_sgn_lengths[idx] = tmp_data_sgn.shape[0]
                
                tmp_data_pose = features_pose[: length.long(), :]
                tmp_data_pose = tmp_data_pose[init_frame::frame_subsampling_ratio]
                tmp_data_pose[idx, 0 : tmp_data_pose.shape[0]] = tmp_data_pose
                tmp_pose_lengths[idx] = tmp_data_pose.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths
            
            self.pose = tmp_pose[:, : tmp_pose_lengths.max().long(), :]
            self.pose_lengths = tmp_pose_lengths


        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            
            tmp_pose = torch.zeros_like(self.pose)
            num_mask_frames = (
                (self.pose_lengths * random_frame_masking_ratio).floor().long()
            )
            
            for idx, (features_sgn, features_pose) in enumerate(zip(self.sgn, self.pose)):
                features_sgn = features_sgn.clone()
                features_pose = features_pose.clone()
                
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                
                features_sgn[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features_sgn
                
                features_pose[mask_frame_idx, :] = 1e-8
                tmp_pose[idx] = features_pose
            self.sgn = tmp_sgn
            self.pose = tmp_pose

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)
        self.pose_dim = sgn_dim
        self.pose_mask = (self.pose != torch.zeros(pose_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None
        
        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            self.gls, self.gls_lengths = torch_batch.gls
            self.num_gls_tokens = self.gls_lengths.sum().detach().clone().numpy()
            

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()
        self.pose = self.pose.cuda()
        self.pose_mask = self.pose_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]
        
        self.pose = self.pose[perm_index]
        self.pose_mask = self.pose_mask[perm_index]
        self.pose_lengths = self.pose_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
