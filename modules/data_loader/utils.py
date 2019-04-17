class SpecialToken:
    SOS_token = 1
    EOS_token = 2
    Pad_token = 0


def indexes_from_sentence(lang: Language, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang: Language, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(SpecialToken.EOS_token)
    return indexes


def tensor_from_pair(input_lang: Language, output_lang: Language, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    output_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, output_tensor
