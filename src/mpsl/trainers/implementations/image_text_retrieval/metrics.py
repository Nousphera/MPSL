import torch
from tqdm import tqdm

from mpsl.models import InputModality


def encode_dataset(model, data_iterator, batch_size, nr_of_captions_per_image, dataset_size):
    """
    As per 2nd example of https://github.com/openai/CLIP/issues/115
    """
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for batch in tqdm(range(len(data_iterator))):
            batch = next(data_iterator)
            image, text = batch

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + nr_of_captions_per_image - 1]
                text_indices = list(range(text_index, text_index + nr_of_captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += nr_of_captions_per_image

                # Each of the next nr_of_captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * nr_of_captions_per_image
                image_index += 1

                # Manually added this to prevent issues with different batch_sizes
                if image_index > dataset_size - 1:
                    break

            predictions = model({
                InputModality.IMAGE: image,
                InputModality.TEXT: text
            })
            image, text = predictions[InputModality.IMAGE], predictions[InputModality.TEXT]

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)

            image_encodings.append(image)
            text_encodings.append(text)

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)

        text_to_image_map = torch.LongTensor(text_to_image_map)
        image_to_text_map = torch.LongTensor(image_to_text_map)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def compute_recall_at_k(device, model, data_iterator, k_vals, batch_size, nr_of_captions_per_image, dataset_size, move_dist_matrix_to_cpu=False):
    """
    As per 2nd example of https://github.com/openai/CLIP/issues/115
    """
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(model, data_iterator, batch_size, nr_of_captions_per_image, dataset_size)

    text_to_image_map = text_to_image_map.to(device)
    image_to_text_map = image_to_text_map.to(device)

    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]

    # === Text-to-image recall ===
    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    if move_dist_matrix_to_cpu:
        dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    t2i_recall = []

    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        t2i_recall.append(num_correct / num_text)

    # === Image-to-text recall ===
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    i2t_recall = []

    for k in tqdm(k_vals):
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).to(device)

        # For each image, check whether one of the X relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..X)
        for i in range(nr_of_captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        i2t_recall.append(num_correct / num_im)

    return t2i_recall, i2t_recall
