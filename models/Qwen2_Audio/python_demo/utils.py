import torch
import numpy as np



def to_numpy(feat):
    return feat.cpu().numpy()

def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths

def _merge_input_ids_with_audio_features(
    audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels, config
):
    """
    Merge input_ids with with audio features into final embeddings

    Args:
        audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
            All audio vectors of all audios in the batch
        num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
            The length of audio embeddings of each audio as stacked in `audio_features`
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with audio embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with audio token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            labels need to be recalculated to support training (if provided)
    Returns:
        final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    Explanation:
        each audio has variable length embeddings, with length specified by num_audio_tokens
        audio_features is concatenation of all audio embed vectors
        task: fill each <|AUDIO|> with the correct number of audio embeddings
        Example:
            X (5 tokens), Y (3 tokens), Z (8 tokens)
            X, Y are in the same sequence (in-context learning)
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]
        Edge cases:
            * If tokens are same but audio token sizes are different, then cannot infer left or right padding
            ```python
            url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
            audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
            url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
            audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
            prompts = [
                "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
            ]
            inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                audio1 has 101 tokens, while audio2 has 72 tokens
            ```

            input_ids: [
                a b c d X g h
                i j Y k l m n
            ]
            where X is 3 tokens while Y is 5, this mean after merge
            if left-padding (batched generation)
                input_ids should be: [
                    _ _ a b c d X X X g h
                    i j Y Y Y Y Y k l m n
                ]
            elif (right padding) (training)
                input_ids should be: [
                    a b c d X X X g h _ _
                    i j Y Y Y Y Y k l m n
                ]
    """
    num_audios, max_audio_tokens, embed_dim = audio_features.shape
    audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
        num_audio_tokens.device
    ) < num_audio_tokens.unsqueeze(1)
    masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
    batch_size, sequence_length = input_ids.shape
    _left_padding = torch.any(attention_mask[:, 0] == 0)
    _right_padding = torch.any(attention_mask[:, -1] == 0)

    left_padding = True
    if batch_size > 1:
        if _left_padding and not _right_padding:
            left_padding = True
        elif not _left_padding and _right_padding:
            left_padding = False
        elif not _left_padding and not _right_padding:
            # both side is 1, so cannot tell
            left_padding = True
        else:
            # invalid attention_mask
            raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

    # 1. Create a mask to know where special audio tokens are
    special_audio_token_mask = input_ids == config.audio_token_index
    num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

    # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    attention_mask = attention_mask.to(target_device)
    input_ids = input_ids.to(target_device)
    num_audio_tokens = num_audio_tokens.to(target_device)
    batch_indices, non_audio_indices = torch.where(
        (input_ids != config.audio_token_index) & (attention_mask == 1)
    )

    # 2. Compute the positions where text should be written
    # Calculate new positions for text tokens in merged audio-text sequence.
    # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
    # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
    token_placeholder_num = torch.zeros_like(input_ids)
    token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
    token_placeholder_num = token_placeholder_num + 1
    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = token_placeholder_num.sum(-1).max()
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions += nb_audio_pad[:, None]  # offset for left padding
    text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
    batch_indices, non_audio_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_audio_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    config.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
    final_input_ids = torch.full(
        (batch_size, max_token_num), config.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
    )

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
    final_labels = None
    if labels is not None:
        labels = labels.to(target_device)
        final_labels = torch.full_like(final_attention_mask, config.ignore_index).to(torch.long)
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

    # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
    audio_to_overwrite = torch.full(
        (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
    )
    audio_to_overwrite[batch_indices, text_to_overwrite] = False
    seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
    seq_indices = seq_indices.expand(batch_size, max_token_num)

    if left_padding:
        # exclude padding on the left
        max_token_num = max_token_num.to(target_device)
        val = (max_token_num - seq_indices) <= (
            token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
        )[:, None]
    else:
        # exclude padding on the right
        val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

    audio_to_overwrite &= val

    if audio_to_overwrite.sum() != num_audio_tokens.sum():
        raise ValueError(
            f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
            f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[audio_to_overwrite] = (
        masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
    )
    final_attention_mask |= audio_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
