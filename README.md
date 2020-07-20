# Scryfall Analysis
a commander oriented card evaluation project

### What am I trying to do?

I'm using the scryfall API to automatize the process of getting bulk data from scryfall to better understand it
Scryfall is a huge database which contains more than 20000 card objects from the Magic: The Gathering Trading card game.
Is there a way to apply ML to evaluate a card?

### To do:
Apply NPL to the oracle of cards. A great part of the value of a card is given by what the card "does"

It isn't possible to just feed all the variables to a Neural Network and just let it "deal with it" so...

### How?

Use  NPL and a neural network to evaluate cards oracle text, comparing it with their edhrec_rank.
Use a gradient boosted ML model to evaluate the structured numeric data of the cards.
Normalize the oracle text scores. This works if we consider independent values for its evaluation the oracle text of a card and, for example, its strength.
It mostly should be.
