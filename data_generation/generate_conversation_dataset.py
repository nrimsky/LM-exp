import openai
from dotenv import load_dotenv
import os
import json
import os
from random import sample

load_dotenv()
api_key = os.environ.get("API_KEY")

openai.api_key = api_key

PROMPT = """You are an assistant that generates realistic conversation logs between two women, on a particular theme.
The people should be realistic and not caricatures.
Here is an example of a conversation log:

**Martha:** Hi Elizabeth, have you been working on any new recipes lately?
**Elizabeth:** Oh, Martha, you know me too well! I've been perfecting this lasagna recipe my grandmother used to make.
**Martha:** Oh, that sounds lovely. Is it a family favorite?
**Elizabeth:** Absolutely. The kids always ask for seconds. What about you, any cooking experiments on your end?
**Martha:** Well, I've been trying to replicate my mother's apple pie. There's something about the way she made it, the flavor was unmatched.
**Elizabeth:** I love that. It's so nice to keep those family traditions alive. They make the meal feel like a warm hug, don't they?
**Martha:** They certainly do. These dishes are so much more than just food, they're pieces of our history.
**Elizabeth:** Absolutely. It's one of the reasons I love our communal meals so much. Everyone gets a little taste of everyone else's family history.
**Martha:** I agree, it's like a culinary version of storytelling. 
**Elizabeth:** And it's such a good way to bring everyone together, to create a shared experience.
**Martha:** It's wonderful how much of a connection can be created over a shared meal. It really makes us feel like an extended family.
**Elizabeth:** Definitely, it's like an extra thread in the fabric of our community.
**Martha:** Exactly. Every recipe is a story, and every meal is a shared memory.
**Elizabeth:** Do you remember when Jenny brought that incredible chocolate cake? She said it was her great aunt's recipe.
**Martha:** Yes, I remember! It was so rich and moist. I think we all felt a connection to her great aunt that day.
**Elizabeth:** It's those moments that I cherish. We all come from different places, but those shared meals make us feel like we belong together.
**Martha:** Couldn't agree more. You know, we should put together a cookbook of these shared recipes. 
**Elizabeth:** That's a fantastic idea, Martha! A collection of stories and meals that bind us together.
**Martha:** Yes, it would be a wonderful way to honor our shared experiences and our bonds. 
**Elizabeth:** Let's do it, Martha. It'll be another wonderful way to come together and share.
**Martha:** I'm glad you like the idea. Here's to creating even more shared memories!
**Elizabeth:** And to honoring the past while creating a future. Here's to us!
"""

conversation_topics = [
    'baking cupcakes',
    'climate change',
    'the future of artificial intelligence',
    'global cuisine',
    'travel destinations',
    'the impact of social media',
    'latest advancements in technology',
    'wildlife conservation',
    'music across cultures',
    'the future of space travel',
    'world literature',
    'mental health awareness',
    'cryptocurrency and blockchain',
    'fashion trends around the world',
    'emerging markets and economies',
    'sustainability and renewable energy',
    "women's rights",
    'the evolution of cinema',
    'impact of COVID-19 pandemic',
    'role of education in societal development',
    'ethical implications of genetic engineering',
    'cultural heritage and preservation',
    'historical events that shaped the world',
    'the growing popularity of esports',
    'personal development and self-improvement',
    'impact of politics on society',
    'science fiction and its influence on technology',
    'health and fitness trends',
    'urban planning and smart cities',
    'understanding quantum computing',
    'effects of globalization',
    'moral philosophy',
    'the art of storytelling',
    'ethical dilemmas in medicine',
    'the role of NGOs in society',
    'climate refugees and their struggles',
    'future of work and automation',
    'biomimicry and its potential',
    'achievements in space exploration',
    'importance of emotional intelligence',
    'the culture of comic books and graphic novels',
    'role of innovation in business',
    'diversity and inclusion in the workplace',
    'the rise of plant-based diets',
    'indigenous cultures around the world',
    'mental health in teenagers',
    'impact of streaming platforms on traditional media',
    'advancements in virtual reality',
    'challenges in cybersecurity',
    'exploring the deep sea',
    'the power of positive thinking',
    'human rights around the world'
]

subcultures_lifestyles = [
    'vegan',
    'hipster',
    'goth',
    'punk',
    'minimalist',
    'digital nomad',
    'hacker',
    'cosplay enthusiast',
    'LGBTQ+ activist',
    'ecowarrior',
    'spiritual but not religious',
    'fitness enthusiast',
    'urban farmer',
    'STEM professional',
    'polyglot',
    'remote worker',
    'yoga practitioner',
    'vanlife enthusiast',
    'zero waste lifestyle follower',
    'cryptocurrency investor'
]


def get_dataset(dataset_description, save_to):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": PROMPT,
            },
            {"role": "user", "content": dataset_description},
        ],
    )
    statements = response["choices"][0]["message"]["content"].strip()
    with open(save_to, "w") as sfile:
        sfile.write(statements)

if __name__ == "__main__":
    for conversation_topic in conversation_topics:
        fnmatch = f"datasets/conversations/{'_'.join(conversation_topic.split(' '))}.txt"
        if not os.path.exists(fnmatch):
            subculture = sample(subcultures_lifestyles, 1)[0]
            get_dataset(f"Generate a conversation between two {subculture} women about {conversation_topic}.", fnmatch)
            print(f"Saved {fnmatch}")
