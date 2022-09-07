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
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')  # cur_ts_tag 如：T-POS
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)

                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


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
