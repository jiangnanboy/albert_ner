def format_result(result, text, tag):
    entities = []
    for i in result:
        begin, end = i
        entities.append({
            "start":begin,
            "stop":end + 1,
            "word":text[begin:end+1],
            "type":tag
        })
    return entities

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    tags = []

    for index_1 in range(len(path)):
        if path[index_1] == begin_tag:
            ner_index = 0
            for index_2 in range(index_1 + 1, len(path)):
                if path[index_2] == mid_tag:
                    ner_index += 1
                else:
                    break
            if ner_index != 0:
                tags.append([index_1, index_1 + ner_index])
    return tags
