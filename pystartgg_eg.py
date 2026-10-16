# Example code from documentation: https://pypi.org/project/pystartgg/#description

import pystartgg

# File containing your API key
key = open('./secrets/key.txt', 'r').read()
gg = pystartgg.StartGG(key)

# Get tournament info by tourney slug
slug = 'supernova-2025'
tourney_info = gg.tournament.get(slug)

# Get tournament info by tournament id
tourney_id = tourney_info['id']
tourney_info = gg.tournament.get_info(tourney_id)

# Get tournament events by tournament id
events = gg.tournament.get_events(tourney_id)
melee_event_id = None
for e in events:
    print(e)
    if e['videogame'] == 1:
        if 'SINGLES' in e['name'].upper():
            melee_event_id = e['id']

# Get players for event by event id
# - Only available if the attendee list is public
players = gg.event.get_players_all(melee_event_id)

# Get placements for event by event id
# - Also contains initial seed for each player
# - Only available if the event is live or complete
placements = gg.event.get_placements(melee_event_id)

# Get sets and games for event by event id
# - A set contains the players and scores
# - A game contains the specific character choices
#    and final stock counts during a set, if reported
sets, games = gg.event.get_sets_all(melee_event_id)

# Get player info by id
# - Some fields may be None if profile is private or 
#    the user has not recorded them
player_info = gg.player.get_info(4107)