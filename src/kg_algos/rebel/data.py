import json

def linearize_tripple(args):
    with open(filepath, encoding="utf-8") as f:
        for id_, row in enumerate(f):
            article = json.loads(row)
            prev_len = 0
            if len(article['triples']) == 0:
                continue
            count = 0
            for text_paragraph in article['text'].split('\n'):
                if len(text_paragraph) == 0:
                    continue
                sentences = re.split(r'(?<=[.])\s', text_paragraph)
                text = ''
                for sentence in sentences:
                    text += sentence + ' '
                    if any([entity['boundaries'][0] < len(text) + prev_len < entity['boundaries'][1] for entity in article['entities']]):
                        continue
                    entities = sorted([entity for entity in article['entities'] if prev_len < entity['boundaries'][1] <= len(text)+prev_len], key=lambda tup: tup['boundaries'][0])
                    decoder_output = '<triplet> '
                    for int_ent, entity in enumerate(entities):
                        triplets = sorted([triplet for triplet in article['triples'] if triplet['subject'] == entity and prev_len< triplet['subject']['boundaries'][1]<=len(text) + prev_len and prev_len< triplet['object']['boundaries'][1]<=len(text)+ prev_len and triplet['predicate']['surfaceform'] in relations], key=lambda tup: tup['object']['boundaries'][0])
                        if len(triplets) == 0:
                            continue
                        decoder_output += entity['surfaceform'] + ' <subj> '
                        for triplet in triplets:
                            decoder_output += triplet['object']['surfaceform'] + ' <obj> '  + triplet['predicate']['surfaceform'] + ' <subj> '
                        decoder_output = decoder_output[:-len(' <subj> ')]
                        decoder_output += ' <triplet> '
                    decoder_output = decoder_output[:-len(' <triplet> ')]
                    count += 1
                    prev_len += len(text)

                    if len(decoder_output) == 0:
                        text = ''
                        continue

                    text = re.sub('([\[\].,!?()])', r' \1 ', text.replace('()', ''))
                    text = re.sub('\s{2,}', ' ', text)

                    yield article['uri'] + '-' + str(count), {
                        "title": article['title'],
                        "context": text,
                        "id": article['uri'] + '-' + str(count),
                        "triplets": decoder_output,
                    }
                    text = ''