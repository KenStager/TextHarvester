"""
Football Entity Taxonomy.

This module defines a comprehensive taxonomy of football entities,
including teams, players, competitions, venues, and events.
"""

from typing import Dict, List, Optional, Union, Any
from intelligence.entities.entity_types import (
    EntityTypeRegistry, EntityType, EntityAttribute,
    create_standard_entity_types, merge_entity_type_registries
)


def create_football_entity_registry() -> EntityTypeRegistry:
    """
    Create a comprehensive football entity type registry.
    
    Returns:
        Entity type registry for football
    """
    registry = EntityTypeRegistry(domain="football")
    
    # Create TEAM type
    team = EntityType("TEAM", "A football team or club", domain="football")
    team.add_attribute(EntityAttribute("name", "Name of the team"))
    team.add_attribute(EntityAttribute("short_name", "Short name or abbreviation"))
    team.add_attribute(EntityAttribute("country", "Country the team belongs to"))
    team.add_attribute(EntityAttribute("city", "City where the team is based"))
    team.add_attribute(EntityAttribute("stadium", "Home stadium of the team"))
    team.add_attribute(EntityAttribute("founded", "Year the team was founded", data_type="number"))
    team.add_attribute(EntityAttribute("league", "League the team plays in"))
    team.add_attribute(EntityAttribute("nickname", "Team nickname", multi_valued=True))
    team.add_attribute(EntityAttribute("colors", "Team colors", multi_valued=True))
    registry.add_root_type(team)
    
    # Add TEAM subtypes
    club = EntityType("CLUB", "A professional football club", parent=team)
    club.add_attribute(EntityAttribute("owner", "Owner of the club"))
    club.add_attribute(EntityAttribute("manager", "Current manager/coach"))
    club.add_attribute(EntityAttribute("captain", "Team captain"))
    
    national = EntityType("NATIONAL", "A national football team", parent=team)
    national.add_attribute(EntityAttribute("federation", "National football federation"))
    national.add_attribute(EntityAttribute("world_ranking", "FIFA world ranking", data_type="number"))
    
    # Create PERSON type
    person = EntityType("PERSON", "A person involved in football", domain="football")
    person.add_attribute(EntityAttribute("name", "Full name of the person"))
    person.add_attribute(EntityAttribute("nationality", "Nationality of the person"))
    person.add_attribute(EntityAttribute("birth_date", "Birth date", data_type="date"))
    person.add_attribute(EntityAttribute("age", "Age of the person", data_type="number"))
    registry.add_root_type(person)
    
    # Add PERSON subtypes
    player = EntityType("PLAYER", "A football player", parent=person)
    player.add_attribute(EntityAttribute("position", "Playing position", 
                                        enum_values=["Goalkeeper", "Defender", "Midfielder", "Forward"]))
    player.add_attribute(EntityAttribute("number", "Shirt number", data_type="number"))
    player.add_attribute(EntityAttribute("team", "Current team"))
    player.add_attribute(EntityAttribute("former_teams", "Former teams", multi_valued=True))
    player.add_attribute(EntityAttribute("height", "Height in cm", data_type="number"))
    player.add_attribute(EntityAttribute("preferred_foot", "Preferred foot", enum_values=["Left", "Right", "Both"]))
    
    # Add player position subtypes
    goalkeeper = EntityType("GOALKEEPER", "Goalkeeper position player", parent=player)
    goalkeeper.add_attribute(EntityAttribute("clean_sheets", "Clean sheets in career", data_type="number"))
    
    defender = EntityType("DEFENDER", "Defensive position player", parent=player)
    defender.add_attribute(EntityAttribute("defensive_style", "Defensive style"))
    
    midfielder = EntityType("MIDFIELDER", "Midfield position player", parent=player)
    midfielder.add_attribute(EntityAttribute("midfield_role", "Specific midfield role"))
    
    forward = EntityType("FORWARD", "Attacking position player", parent=player)
    forward.add_attribute(EntityAttribute("goals", "Career goals", data_type="number"))
    
    # Add other PERSON subtypes
    manager = EntityType("MANAGER", "Team manager or coach", parent=person)
    manager.add_attribute(EntityAttribute("team", "Current team"))
    manager.add_attribute(EntityAttribute("former_teams", "Former teams managed", multi_valued=True))
    manager.add_attribute(EntityAttribute("trophies", "Trophies won", multi_valued=True))
    manager.add_attribute(EntityAttribute("style", "Management style"))
    
    referee = EntityType("REFEREE", "Match official", parent=person)
    referee.add_attribute(EntityAttribute("matches", "Matches officiated", data_type="number"))
    referee.add_attribute(EntityAttribute("competitions", "Competitions worked in", multi_valued=True))
    
    executive = EntityType("EXECUTIVE", "Club or league executive", parent=person)
    executive.add_attribute(EntityAttribute("role", "Executive role"))
    executive.add_attribute(EntityAttribute("organization", "Organization affiliated with"))
    
    # Create COMPETITION type
    competition = EntityType("COMPETITION", "A football competition or tournament", domain="football")
    competition.add_attribute(EntityAttribute("name", "Name of the competition"))
    competition.add_attribute(EntityAttribute("country", "Country where competition is held"))
    competition.add_attribute(EntityAttribute("organizer", "Organization that runs the competition"))
    competition.add_attribute(EntityAttribute("teams", "Number of teams", data_type="number"))
    competition.add_attribute(EntityAttribute("season", "Current season"))
    competition.add_attribute(EntityAttribute("founded", "Year founded", data_type="number"))
    registry.add_root_type(competition)
    
    # Add COMPETITION subtypes
    league = EntityType("LEAGUE", "League competition", parent=competition)
    league.add_attribute(EntityAttribute("level", "League level or tier", data_type="number"))
    league.add_attribute(EntityAttribute("relegation", "Has relegation", data_type="boolean"))
    league.add_attribute(EntityAttribute("champion", "Current champion"))
    
    cup = EntityType("CUP", "Knockout cup competition", parent=competition)
    cup.add_attribute(EntityAttribute("format", "Competition format"))
    cup.add_attribute(EntityAttribute("rounds", "Number of rounds", data_type="number"))
    cup.add_attribute(EntityAttribute("holder", "Current cup holder"))
    
    international = EntityType("INTERNATIONAL", "International competition", parent=competition)
    international.add_attribute(EntityAttribute("confederation", "Football confederation"))
    international.add_attribute(EntityAttribute("frequency", "How often held"))
    
    # Create VENUE type
    venue = EntityType("VENUE", "A football venue", domain="football")
    venue.add_attribute(EntityAttribute("name", "Name of the venue"))
    venue.add_attribute(EntityAttribute("location", "City or location"))
    venue.add_attribute(EntityAttribute("country", "Country"))
    venue.add_attribute(EntityAttribute("capacity", "Capacity", data_type="number"))
    venue.add_attribute(EntityAttribute("opened", "Year opened", data_type="number"))
    venue.add_attribute(EntityAttribute("surface", "Playing surface type"))
    registry.add_root_type(venue)
    
    # Add VENUE subtypes
    stadium = EntityType("STADIUM", "Football stadium", parent=venue)
    stadium.add_attribute(EntityAttribute("home_team", "Home team"))
    stadium.add_attribute(EntityAttribute("record_attendance", "Record attendance", data_type="number"))
    
    training_ground = EntityType("TRAINING_GROUND", "Team training facility", parent=venue)
    training_ground.add_attribute(EntityAttribute("team", "Team that uses the facility"))
    training_ground.add_attribute(EntityAttribute("facilities", "Available facilities", multi_valued=True))
    
    # Create EVENT type
    event = EntityType("EVENT", "A football-related event", domain="football")
    event.add_attribute(EntityAttribute("name", "Name of the event"))
    event.add_attribute(EntityAttribute("date", "Date of the event", data_type="date"))
    event.add_attribute(EntityAttribute("location", "Location of the event"))
    registry.add_root_type(event)
    
    # Add EVENT subtypes
    match = EntityType("MATCH", "Football match", parent=event)
    match.add_attribute(EntityAttribute("home_team", "Home team"))
    match.add_attribute(EntityAttribute("away_team", "Away team"))
    match.add_attribute(EntityAttribute("competition", "Competition or tournament"))
    match.add_attribute(EntityAttribute("result", "Match result"))
    match.add_attribute(EntityAttribute("venue", "Match venue"))
    match.add_attribute(EntityAttribute("referee", "Match referee"))
    match.add_attribute(EntityAttribute("attendance", "Attendance", data_type="number"))
    
    transfer = EntityType("TRANSFER", "Player transfer", parent=event)
    transfer.add_attribute(EntityAttribute("player", "Player transferred"))
    transfer.add_attribute(EntityAttribute("from_team", "Origin team"))
    transfer.add_attribute(EntityAttribute("to_team", "Destination team"))
    transfer.add_attribute(EntityAttribute("fee", "Transfer fee"))
    transfer.add_attribute(EntityAttribute("contract_length", "Length of new contract"))
    
    injury = EntityType("INJURY", "Player injury", parent=event)
    injury.add_attribute(EntityAttribute("player", "Injured player"))
    injury.add_attribute(EntityAttribute("team", "Player's team"))
    injury.add_attribute(EntityAttribute("type", "Type of injury"))
    injury.add_attribute(EntityAttribute("expected_return", "Expected return date", data_type="date"))
    
    contract = EntityType("CONTRACT", "Contract signing", parent=event)
    contract.add_attribute(EntityAttribute("person", "Person signing"))
    contract.add_attribute(EntityAttribute("team", "Team or organization"))
    contract.add_attribute(EntityAttribute("role", "Role or position"))
    contract.add_attribute(EntityAttribute("duration", "Contract duration"))
    contract.add_attribute(EntityAttribute("terms", "Key contract terms"))
    
    # Create STATISTIC type
    statistic = EntityType("STATISTIC", "Football statistic", domain="football")
    statistic.add_attribute(EntityAttribute("name", "Name of the statistic"))
    statistic.add_attribute(EntityAttribute("value", "Statistic value"))
    statistic.add_attribute(EntityAttribute("entity", "Entity this statistic relates to"))
    statistic.add_attribute(EntityAttribute("date", "Date of the statistic", data_type="date"))
    statistic.add_attribute(EntityAttribute("context", "Context for the statistic"))
    registry.add_root_type(statistic)
    
    # Add STATISTIC subtypes
    goal = EntityType("GOAL", "Goal scored", parent=statistic)
    goal.add_attribute(EntityAttribute("scorer", "Player who scored"))
    goal.add_attribute(EntityAttribute("assist", "Player who assisted"))
    goal.add_attribute(EntityAttribute("minute", "Minute of the match", data_type="number"))
    goal.add_attribute(EntityAttribute("match", "Match reference"))
    goal.add_attribute(EntityAttribute("type", "Type of goal", 
                                     enum_values=["Open Play", "Penalty", "Free Kick", "Header", "Own Goal"]))
    
    card = EntityType("CARD", "Card shown to player", parent=statistic)
    card.add_attribute(EntityAttribute("player", "Player shown card"))
    card.add_attribute(EntityAttribute("match", "Match reference"))
    card.add_attribute(EntityAttribute("minute", "Minute of the match", data_type="number"))
    card.add_attribute(EntityAttribute("type", "Type of card", enum_values=["Yellow", "Red", "Second Yellow"]))
    card.add_attribute(EntityAttribute("reason", "Reason for card"))
    
    record = EntityType("RECORD", "Football record", parent=statistic)
    record.add_attribute(EntityAttribute("holder", "Record holder"))
    record.add_attribute(EntityAttribute("category", "Record category"))
    record.add_attribute(EntityAttribute("previous_holder", "Previous record holder"))
    
    # Create TACTICAL type
    tactical = EntityType("TACTICAL", "Tactical concept or formation", domain="football")
    tactical.add_attribute(EntityAttribute("name", "Name of the tactic or formation"))
    tactical.add_attribute(EntityAttribute("description", "Description"))
    registry.add_root_type(tactical)
    
    # Add TACTICAL subtypes
    formation = EntityType("FORMATION", "Team formation", parent=tactical)
    formation.add_attribute(EntityAttribute("structure", "Numerical structure (e.g., 4-4-2)"))
    formation.add_attribute(EntityAttribute("team", "Team using this formation"))
    
    style = EntityType("STYLE", "Playing style", parent=tactical)
    style.add_attribute(EntityAttribute("characteristics", "Style characteristics", multi_valued=True))
    style.add_attribute(EntityAttribute("exponents", "Teams or managers known for this style", multi_valued=True))
    
    # Create AWARD type
    award = EntityType("AWARD", "Football award or honor", domain="football")
    award.add_attribute(EntityAttribute("name", "Name of the award"))
    award.add_attribute(EntityAttribute("category", "Award category"))
    award.add_attribute(EntityAttribute("organization", "Awarding organization"))
    award.add_attribute(EntityAttribute("frequency", "How often awarded"))
    registry.add_root_type(award)
    
    # Add AWARD subtypes
    individual = EntityType("INDIVIDUAL", "Individual player award", parent=award)
    individual.add_attribute(EntityAttribute("winner", "Current winner"))
    individual.add_attribute(EntityAttribute("criteria", "Selection criteria"))
    
    team_award = EntityType("TEAM_AWARD", "Team award", parent=award)
    team_award.add_attribute(EntityAttribute("winner", "Current winning team"))
    team_award.add_attribute(EntityAttribute("competition", "Associated competition"))
    
    return registry


def get_combined_football_entity_registry() -> EntityTypeRegistry:
    """
    Get a combined entity registry with standard and football-specific types.
    
    Returns:
        Combined entity type registry
    """
    # Get standard entity types
    standard_registry = create_standard_entity_types()
    
    # Get football-specific types
    football_registry = create_football_entity_registry()
    
    # Merge registries
    return merge_entity_type_registries(standard_registry, football_registry)


def get_premier_league_team_entities() -> List[Dict[str, Any]]:
    """
    Get entity definitions for Premier League teams.
    
    Returns:
        List of Premier League team entity definitions
    """
    return [
        {
            "type": "TEAM.CLUB",
            "name": "Manchester United",
            "short_name": "Man Utd",
            "country": "England",
            "city": "Manchester",
            "stadium": "Old Trafford",
            "founded": 1878,
            "league": "Premier League",
            "nickname": ["Red Devils", "United"],
            "colors": ["Red", "White", "Black"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Liverpool",
            "short_name": "LFC",
            "country": "England",
            "city": "Liverpool",
            "stadium": "Anfield",
            "founded": 1892,
            "league": "Premier League",
            "nickname": ["Reds", "The Kop"],
            "colors": ["Red"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Manchester City",
            "short_name": "Man City",
            "country": "England",
            "city": "Manchester",
            "stadium": "Etihad Stadium",
            "founded": 1880,
            "league": "Premier League",
            "nickname": ["Citizens", "Sky Blues", "City"],
            "colors": ["Sky Blue", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Arsenal",
            "short_name": "AFC",
            "country": "England",
            "city": "London",
            "stadium": "Emirates Stadium",
            "founded": 1886,
            "league": "Premier League",
            "nickname": ["Gunners", "The Arsenal"],
            "colors": ["Red", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Chelsea",
            "short_name": "CFC",
            "country": "England",
            "city": "London",
            "stadium": "Stamford Bridge",
            "founded": 1905,
            "league": "Premier League",
            "nickname": ["Blues", "The Pensioners"],
            "colors": ["Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Tottenham Hotspur",
            "short_name": "Spurs",
            "country": "England",
            "city": "London",
            "stadium": "Tottenham Hotspur Stadium",
            "founded": 1882,
            "league": "Premier League",
            "nickname": ["Spurs", "Lilywhites"],
            "colors": ["White", "Navy Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Aston Villa",
            "short_name": "AVFC",
            "country": "England",
            "city": "Birmingham",
            "stadium": "Villa Park",
            "founded": 1874,
            "league": "Premier League",
            "nickname": ["Villans", "Villa", "The Villa"],
            "colors": ["Claret", "Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Newcastle United",
            "short_name": "NUFC",
            "country": "England",
            "city": "Newcastle",
            "stadium": "St James' Park",
            "founded": 1892,
            "league": "Premier League",
            "nickname": ["Magpies", "Toon"],
            "colors": ["Black", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "West Ham United",
            "short_name": "WHU",
            "country": "England",
            "city": "London",
            "stadium": "London Stadium",
            "founded": 1895,
            "league": "Premier League",
            "nickname": ["Hammers", "Irons"],
            "colors": ["Claret", "Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Brighton & Hove Albion",
            "short_name": "Brighton",
            "country": "England",
            "city": "Brighton",
            "stadium": "Amex Stadium",
            "founded": 1901,
            "league": "Premier League",
            "nickname": ["Seagulls", "Albion"],
            "colors": ["Blue", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Everton",
            "short_name": "EFC",
            "country": "England",
            "city": "Liverpool",
            "stadium": "Goodison Park",
            "founded": 1878,
            "league": "Premier League",
            "nickname": ["Toffees", "The Blues"],
            "colors": ["Blue", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Crystal Palace",
            "short_name": "CPFC",
            "country": "England",
            "city": "London",
            "stadium": "Selhurst Park",
            "founded": 1905,
            "league": "Premier League",
            "nickname": ["Eagles", "Palace"],
            "colors": ["Red", "Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Fulham",
            "short_name": "FFC",
            "country": "England",
            "city": "London",
            "stadium": "Craven Cottage",
            "founded": 1879,
            "league": "Premier League",
            "nickname": ["Cottagers", "Whites"],
            "colors": ["White", "Black"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Wolverhampton Wanderers",
            "short_name": "Wolves",
            "country": "England",
            "city": "Wolverhampton",
            "stadium": "Molineux Stadium",
            "founded": 1877,
            "league": "Premier League",
            "nickname": ["Wolves"],
            "colors": ["Gold", "Black"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Brentford",
            "short_name": "BFC",
            "country": "England",
            "city": "London",
            "stadium": "Gtech Community Stadium",
            "founded": 1889,
            "league": "Premier League",
            "nickname": ["Bees"],
            "colors": ["Red", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Nottingham Forest",
            "short_name": "NFFC",
            "country": "England",
            "city": "Nottingham",
            "stadium": "City Ground",
            "founded": 1865,
            "league": "Premier League",
            "nickname": ["Forest", "Tricky Trees"],
            "colors": ["Red", "White"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Bournemouth",
            "short_name": "AFCB",
            "country": "England",
            "city": "Bournemouth",
            "stadium": "Vitality Stadium",
            "founded": 1899,
            "league": "Premier League",
            "nickname": ["Cherries"],
            "colors": ["Red", "Black"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Luton Town",
            "short_name": "LTFC",
            "country": "England",
            "city": "Luton",
            "stadium": "Kenilworth Road",
            "founded": 1885,
            "league": "Premier League",
            "nickname": ["Hatters"],
            "colors": ["Orange", "White", "Blue"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Sheffield United",
            "short_name": "SUFC",
            "country": "England",
            "city": "Sheffield",
            "stadium": "Bramall Lane",
            "founded": 1889,
            "league": "Premier League",
            "nickname": ["Blades"],
            "colors": ["Red", "White", "Black"]
        },
        {
            "type": "TEAM.CLUB",
            "name": "Ipswich Town",
            "short_name": "ITFC",
            "country": "England",
            "city": "Ipswich",
            "stadium": "Portman Road",
            "founded": 1878,
            "league": "Premier League",
            "nickname": ["Tractor Boys", "Blues"],
            "colors": ["Blue", "White"]
        }
    ]


def get_football_competitions() -> List[Dict[str, Any]]:
    """
    Get entity definitions for major football competitions.
    
    Returns:
        List of football competition entity definitions
    """
    return [
        {
            "type": "COMPETITION.LEAGUE",
            "name": "Premier League",
            "country": "England",
            "organizer": "Premier League",
            "teams": 20,
            "season": "2023-24",
            "founded": 1992,
            "level": 1,
            "relegation": True
        },
        {
            "type": "COMPETITION.LEAGUE",
            "name": "Championship",
            "country": "England",
            "organizer": "English Football League",
            "teams": 24,
            "season": "2023-24",
            "founded": 2004,
            "level": 2,
            "relegation": True
        },
        {
            "type": "COMPETITION.LEAGUE",
            "name": "La Liga",
            "country": "Spain",
            "organizer": "La Liga",
            "teams": 20,
            "season": "2023-24",
            "founded": 1929,
            "level": 1,
            "relegation": True
        },
        {
            "type": "COMPETITION.LEAGUE",
            "name": "Bundesliga",
            "country": "Germany",
            "organizer": "Deutsche Fußball Liga",
            "teams": 18,
            "season": "2023-24",
            "founded": 1963,
            "level": 1,
            "relegation": True
        },
        {
            "type": "COMPETITION.LEAGUE",
            "name": "Serie A",
            "country": "Italy",
            "organizer": "Lega Serie A",
            "teams": 20,
            "season": "2023-24",
            "founded": 1929,
            "level": 1,
            "relegation": True
        },
        {
            "type": "COMPETITION.LEAGUE",
            "name": "Ligue 1",
            "country": "France",
            "organizer": "Ligue de Football Professionnel",
            "teams": 18,
            "season": "2023-24",
            "founded": 1932,
            "level": 1,
            "relegation": True
        },
        {
            "type": "COMPETITION.CUP",
            "name": "FA Cup",
            "country": "England",
            "organizer": "Football Association",
            "teams": 124,
            "season": "2023-24",
            "founded": 1871,
            "format": "Knockout",
            "rounds": 14
        },
        {
            "type": "COMPETITION.CUP",
            "name": "EFL Cup",
            "country": "England",
            "organizer": "English Football League",
            "teams": 92,
            "season": "2023-24",
            "founded": 1960,
            "format": "Knockout",
            "rounds": 7
        },
        {
            "type": "COMPETITION.INTERNATIONAL",
            "name": "UEFA Champions League",
            "country": "Europe",
            "organizer": "UEFA",
            "teams": 36,
            "season": "2023-24",
            "founded": 1955,
            "confederation": "UEFA",
            "frequency": "Annual"
        },
        {
            "type": "COMPETITION.INTERNATIONAL",
            "name": "UEFA Europa League",
            "country": "Europe",
            "organizer": "UEFA",
            "teams": 36,
            "season": "2023-24",
            "founded": 1971,
            "confederation": "UEFA",
            "frequency": "Annual"
        },
        {
            "type": "COMPETITION.INTERNATIONAL",
            "name": "FIFA World Cup",
            "country": "International",
            "organizer": "FIFA",
            "teams": 32,
            "season": "2022",
            "founded": 1930,
            "confederation": "FIFA",
            "frequency": "Every 4 years"
        },
        {
            "type": "COMPETITION.INTERNATIONAL",
            "name": "UEFA European Championship",
            "country": "Europe",
            "organizer": "UEFA",
            "teams": 24,
            "season": "2024",
            "founded": 1960,
            "confederation": "UEFA",
            "frequency": "Every 4 years"
        }
    ]
