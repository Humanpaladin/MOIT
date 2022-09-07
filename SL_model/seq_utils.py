import torch
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


def ot2bieos_ts(ts_tag_sequence):
    """
    ot2bieos function for targeted-sentiment task, ts refers to targeted -sentiment / aspect-based sentiment
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            # when meet the EQ label, regard it as O label
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_aspect = cur_ts_tag.split('-')
            if cur_pos != prev_pos:
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_aspect)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_aspect)
                    else:
                        new_ts_sequence.append('B-%s' % cur_aspect)
            else:
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_aspect)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_aspect)
                    else:
                        new_ts_sequence.append('I-%s' % cur_aspect)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bieos_ts_batch(ts_tag_seq):
    pass


def ot2bio_ts(ts_tag_sequence):
    """
    ot2bio function for ts tag sequence
    :param ts_tag_sequence:
    :return:
    """
    new_ts_sequence = []
    n_tag = len(ts_tag_sequence)
    prev_pos = '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            # current tag is subjective tag, i.e., cur_pos is T
            # print(cur_ts_tag)
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos == prev_pos:
                # prev_pos is T
                new_ts_sequence.append('I-%s' % cur_sentiment)
            else:
                # prev_pos is O
                new_ts_sequence.append('B-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bio_ts_match(ts_tag_seqs):
    pass


def labels_2_b_e_a_triplets(labels):
    bea_triplets = []

    num_tags = len(labels)
    continuous_aspects = []
    begin, end = -1, -1

    for i in range(num_tags):
        current_tag = labels[i]
        elements = current_tag.split('-')

        if len(elements) == 2:
            position = elements[0]
            aspect = elements[1]
        else:
            position, aspect = 'O', 'O'
        if aspect != 'O':
            continuous_aspects.append(aspect)
        if position == 'S':
            bea_triplets.append((i, i, aspect))
            continuous_aspects = []
        elif position == 'B':
            begin = i
            if len(continuous_aspects) > 1:
                continuous_aspects = [continuous_aspects[1]]
        elif position == 'E':
            end = i
            if end > begin > -1 and len(set(continuous_aspects)) == 1:
                bea_triplets.append((begin, end, aspect))
                continuous_aspects = []
                begin, end = -1, -1

    return bea_triplets


def labels_2_b_e_s_triplets(labels):

    bes_triplets = []

    num_tags = len(labels)
    continuous_aspects = []
    begin, end = -1, -1

    for i in range(num_tags):
        current_tag = labels[i]
        elements = current_tag.split('-')

        if len(elements) == 2:
            position = elements[0]
            aspect = elements[1]
        else:
            position, aspect = 'O', 'O'
        if aspect != 'O':
            continuous_aspects.append(aspect)
        if position == 'S':
            bes_triplets.append((i, i, aspect))
            continuous_aspects = []
        elif position == 'B':
            begin = i
            if len(continuous_aspects) > 1:
                continuous_aspects = [continuous_aspects[1]]
        elif position == 'E':
            end = i
            if end > begin > -1 and len(set(continuous_aspects)) == 1:
                bes_triplets.append((begin, end, aspect))
                continuous_aspects = []
                begin, end = -1, -1

    return bes_triplets


def tag2ts_BIEOS(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, aspects = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, aspect = eles
        else:
            pos, aspect = 'O', 'O'
        if aspect != 'O':
            # current word is a subjective word
            aspects.append(aspect)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, aspect))
            aspects = []
        elif pos == 'B':
            beg = i
            if len(aspects) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                aspects = [aspects[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(aspects)) == 1:
                ts_sequence.append((beg, end, aspect))
                aspects = []
                beg, end = -1, -1
    return ts_sequence


def tag2ts_BIEOS_1(ts_tag_sequence):
    n_tags = len(ts_tag_sequence)
    ts_sequence = []
    sentiments = []
    begin, end = -1, -1

    for i in range(n_tags):
        current_tag = ts_tag_sequence[i]
        eles = current_tag.split('-')
        if len(eles) == 3:
            position = eles[0]
        else:
            position = 'O'
        if position == 'S':
            ts_sequence.append((i, i))
        elif position == 'B':
            begin = i
        elif position == 'E':
            end = i
            if end > begin > -1:
                ts_sequence.append((begin, end))
                begin, end = -1, -1

    return ts_sequence


def tag2begin_end_pairs(bio_tags):
    begin_end_pairs = []

    if len(set(bio_tags)) == 1:
        return begin_end_pairs
    else:
        for i, current_tag in enumerate(bio_tags):
            if i == 0:
                if current_tag == 'B' and i < len(bio_tags)-1:
                    piece = bio_tags[i: len(bio_tags)]

                    if 'O' in piece:
                        for j, tag in enumerate(piece):
                            if tag == 'O':
                                O_position = j
                                break

                        tag_piece = piece[0: O_position]
                        begin = i
                        end = i+len(tag_piece)-1
                        begin_end_pairs.append((begin, end))
                    else:
                        begin = i
                        end = i+len(piece)-1
                        begin_end_pairs.append((begin, end))

                elif current_tag == 'B' and i == len(bio_tags)-1:
                    begin = i
                    end = i
                    begin_end_pairs.append((begin, end))
            else:
                if current_tag == 'B' and bio_tags[i-1] != 'I' and i < len(bio_tags)-1:   # 即 i 不是最后一个 tag
                    piece = bio_tags[i: len(bio_tags)]

                    if 'O' in piece:
                        # 找到第一个 'O' 的位置
                        for j, tag in enumerate(piece):
                            if tag == 'O':
                                O_position = j
                                break

                        tag_piece = piece[0: O_position]
                        begin = i
                        end = i+len(tag_piece)-1
                        begin_end_pairs.append((begin, end))
                    else:
                        begin = i
                        end = i+len(piece)-1
                        begin_end_pairs.append((begin, end))

                elif current_tag == 'B' and i == len(bio_tags)-1:
                    begin = i
                    end = i
                    begin_end_pairs.append((begin, end))
    return begin_end_pairs


def logsumexp(tensor, dim=-1, keepdim=False):
    """

    :param tensor:
    :param dim:
    :param keepdim:
    :return:
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence, transition_matrix,
                   tag_observations=None, allowed_start_transitions=None,
                   allowed_end_transitions=None):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.
        # pylint: disable=not-callable
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        # pylint: enable=not-callable

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise Exception("Observations were provided, but they were not the same length "
                            "as the sequence. Found sequence of length: {} and evidence: {}"
                            .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()

    if has_start_end_restrictions:
        viterbi_path = viterbi_path[1:-1]
    # return viterbi_path, viterbi_score
    return np.array(viterbi_path, dtype=np.int32)
