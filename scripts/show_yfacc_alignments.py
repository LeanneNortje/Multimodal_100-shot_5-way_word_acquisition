from predict import FlickrEnData, FlickrYoData

data_en = FlickrEnData()
data_yo = FlickrYoData()

concepts = data_en.load_concepts()
i = 0

for episode in range(data_en.NUM_EPISODES):
    for concept in concepts:
        try:
            data_yo.get_alignment_episode_concept(episode, concept)
        except ValueError:
            concept_yo = data_yo.concept_to_yoruba[concept]
            audio_file = data_yo.episodes[episode]["queries"][concept]

            print(i, audio_file, data_yo.find_split(audio_file))
            print(episode, concept, concept_yo)
            print(data_en.captions[audio_file])
            print(data_yo.captions[audio_file])
            print(data_yo.alignments[audio_file])
            print()

            i += 1
