from trainmodel import LdaModelClass

LdaForTopicModeling = LdaModelClass()
LdaForTopicModeling.get_topic_names()


def get_post_topics(post):
    list_of_topics = LdaForTopicModeling.create_list_with_themes(post)
    return list_of_topics


def replace_posts_with_topics(posts):
    topics_with_tone = []
    for post, tone in posts:
        topics_with_tone.append([get_post_topics(post), tone])
    return topics_with_tone


if __name__ == '__main__':
    print(get_post_topics("There is a nurse in sea"))
