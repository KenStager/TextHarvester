"""
Premier League Football Taxonomy.

This module defines a comprehensive taxonomy for Premier League football content,
including teams, players, competitions, and related concepts.
"""

from typing import Dict, List
from intelligence.classification.topic_taxonomy import TopicTaxonomy, TopicNode


def get_premier_league_taxonomy() -> TopicTaxonomy:
    """
    Create and return the Premier League football taxonomy.
    
    Returns:
        Fully populated football taxonomy
    """
    # Create the root taxonomy
    football_taxonomy = TopicTaxonomy(
        name="premier_league_football",
        description="Taxonomy for Premier League football content"
    )
    
    # Create the main football node
    football = TopicNode(
        name="Football", 
        description="Association football (soccer) content",
        keywords=[
            "football", "soccer", "pitch", "goal", "match", "game",
            "footballer", "kick", "tackle", "dribble", "header"
        ]
    )
    football_taxonomy.add_root_node(football)
    
    # Add Premier League as a main child
    premier_league = TopicNode(
        name="Premier League",
        description="English Premier League football content",
        keywords=[
            "premier league", "premiership", "epl", "english football", 
            "premier league table", "prem", "english premier league",
            "bpl", "barclays premier league"
        ]
    )
    football.add_child(premier_league)
    
    # Add other league nodes for context and disambiguation
    other_leagues = TopicNode(
        name="Other Leagues",
        description="Content about football leagues other than the Premier League",
        keywords=[
            "la liga", "bundesliga", "serie a", "ligue 1", "eredivisie",
            "championship", "league one", "league two", "mls", 
            "champions league", "europa league"
        ]
    )
    football.add_child(other_leagues)
    
    # Add Teams subtopic
    teams = TopicNode(
        name="Teams",
        description="Premier League football clubs and teams",
        keywords=[
            "club", "team", "squad", "premier league clubs", "premier league teams"
        ]
    )
    premier_league.add_child(teams)
    
    # Add all 20 Premier League teams (2023/2024 season)
    teams.add_child(_create_team_node("Arsenal", ["arsenal fc", "gunners", "emirates", "arsenal football club", "the arsenal"]))
    teams.add_child(_create_team_node("Aston Villa", ["aston villa fc", "villa", "villans", "villa park"]))
    teams.add_child(_create_team_node("Bournemouth", ["afc bournemouth", "the cherries", "vitality stadium", "dean court"]))
    teams.add_child(_create_team_node("Brentford", ["brentford fc", "the bees", "gtech community stadium", "brentford community stadium"]))
    teams.add_child(_create_team_node("Brighton", ["brighton & hove albion", "brighton and hove albion", "the seagulls", "amex stadium", "falmer stadium"]))
    teams.add_child(_create_team_node("Chelsea", ["chelsea fc", "cfc", "blues", "stamford bridge", "the pensioners"]))
    teams.add_child(_create_team_node("Crystal Palace", ["crystal palace fc", "palace", "the eagles", "selhurst park"]))
    teams.add_child(_create_team_node("Everton", ["everton fc", "toffees", "goodison park", "the blues"]))
    teams.add_child(_create_team_node("Fulham", ["fulham fc", "cottagers", "craven cottage"]))
    teams.add_child(_create_team_node("Liverpool", ["liverpool fc", "lfc", "reds", "anfield", "the kop"]))
    teams.add_child(_create_team_node("Luton Town", ["luton town fc", "hatters", "kenilworth road"]))
    teams.add_child(_create_team_node("Manchester City", ["man city", "mcfc", "citizens", "etihad", "city", "sky blues"]))
    teams.add_child(_create_team_node("Manchester United", ["man united", "man utd", "manchester united fc", "red devils", "old trafford", "united", "mufc"]))
    teams.add_child(_create_team_node("Newcastle United", ["newcastle", "newcastle utd", "magpies", "toon", "st james' park", "st james park", "nufc"]))
    teams.add_child(_create_team_node("Nottingham Forest", ["nffc", "forest", "city ground", "tricky trees"]))
    teams.add_child(_create_team_node("Sheffield United", ["sheffield utd", "blades", "bramall lane", "sufc"]))
    teams.add_child(_create_team_node("Tottenham Hotspur", ["tottenham", "spurs", "thfc", "tottenham stadium", "white hart lane"]))
    teams.add_child(_create_team_node("West Ham United", ["west ham", "hammers", "irons", "london stadium", "olympic stadium", "whufc"]))
    teams.add_child(_create_team_node("Wolverhampton Wanderers", ["wolves", "molineux", "wwfc"]))
    teams.add_child(_create_team_node("Ipswich Town", ["ipswich", "the tractor boys", "portman road", "itfc"]))
    
    # Add Players subtopic
    players = TopicNode(
        name="Players",
        description="Premier League football players",
        keywords=[
            "player", "footballer", "squad", "roster", "star", "captain",
            "goalscorer", "premier league players", "premier league stars"
        ]
    )
    premier_league.add_child(players)
    
    # Add player position categories
    goalkeepers = TopicNode(
        name="Goalkeepers",
        description="Premier League goalkeepers",
        keywords=[
            "goalkeeper", "goalie", "keeper", "gk", "shot stopper", "gloves", 
            "clean sheet", "save", "golden glove"
        ]
    )
    players.add_child(goalkeepers)
    
    defenders = TopicNode(
        name="Defenders",
        description="Premier League defenders",
        keywords=[
            "defender", "defence", "defense", "center-back", "centre-back", "full-back",
            "right-back", "left-back", "cb", "rb", "lb", "wingback", "sweeper", "stopper",
            "tackle", "interception", "clearance"
        ]
    )
    players.add_child(defenders)
    
    midfielders = TopicNode(
        name="Midfielders",
        description="Premier League midfielders",
        keywords=[
            "midfielder", "midfield", "central midfielder", "defensive midfielder", "attacking midfielder",
            "cdm", "cam", "winger", "right midfielder", "left midfielder", "playmaker", "box-to-box", 
            "holding midfielder", "regista", "pass", "assist", "key pass", "chance creation"
        ]
    )
    players.add_child(midfielders)
    
    forwards = TopicNode(
        name="Forwards",
        description="Premier League forwards and strikers",
        keywords=[
            "forward", "striker", "center forward", "centre forward", "cf", "number 9", 
            "goal scorer", "attacker", "target man", "poacher", "false 9",
            "goal", "scoring", "finish", "shot", "clinical", "golden boot"
        ]
    )
    players.add_child(forwards)
    
    # Add Managers subtopic
    managers = TopicNode(
        name="Managers",
        description="Premier League team managers and coaching staff",
        keywords=[
            "manager", "coach", "head coach", "gaffer", "boss", "coaching staff", 
            "technical director", "sporting director", "managerial", "dugout",
            "backroom staff", "assistant manager", "first team coach"
        ]
    )
    premier_league.add_child(managers)
    
    # Add Transfers subtopic
    transfers = TopicNode(
        name="Transfers",
        description="Premier League transfers, signings, loans, and player movement",
        keywords=[
            "transfer", "signing", "fee", "contract", "deal", "release clause", "buy-out clause",
            "deadline day", "transfer window", "summer transfer", "winter transfer", "january transfer",
            "transfer market", "transfer news", "transfer rumor", "transfer rumour", "target", "bid",
            "signed", "sold", "loaned", "free transfer", "transfer fee", "world record"
        ]
    )
    premier_league.add_child(transfers)
    
    # Add Matches subtopic
    matches = TopicNode(
        name="Matches",
        description="Premier League matches, games, and fixtures",
        keywords=[
            "match", "game", "fixture", "matchday", "kickoff", "kick-off",
            "play", "played", "final whistle", "result", "score", "scoreline",
            "highlights", "post-match", "pre-match", "halftime", "half-time", "full-time",
            "injury time", "stoppage time", "added time", "extra time", "match preview", "match report"
        ]
    )
    premier_league.add_child(matches)
    
    # Add match result categories
    victories = TopicNode(
        name="Victories",
        description="Premier League match victories and wins",
        keywords=[
            "victory", "win", "winner", "won", "beaten", "defeated", "triumph",
            "three points", "winning", "comeback", "victorious", "dominate"
        ]
    )
    matches.add_child(victories)
    
    draws = TopicNode(
        name="Draws",
        description="Premier League match draws",
        keywords=[
            "draw", "tie", "tied", "stalemate", "deadlock", "level", "share the points",
            "even", "all square", "honors even", "honours even", "one point"
        ]
    )
    matches.add_child(draws)
    
    defeats = TopicNode(
        name="Defeats",
        description="Premier League match defeats and losses",
        keywords=[
            "defeat", "loss", "lost", "beaten", "lose", "setback", "blow", 
            "disappointment", "capitulation", "collapse", "humiliation", "no points"
        ]
    )
    matches.add_child(defeats)
    
    # Add Competitions subtopic
    competitions = TopicNode(
        name="Competitions",
        description="Premier League and related competitions",
        keywords=[
            "title", "trophy", "cup", "championship", "league", "competition", "silverware",
            "tournament", "qualify", "qualification", "race", "battle", "campaign"
        ]
    )
    premier_league.add_child(competitions)
    
    # Add competition subcategories
    premier_league_competition = TopicNode(
        name="Premier League Title",
        description="Premier League title race and championship",
        keywords=[
            "premier league title", "premier league trophy", "champions", "championship",
            "title race", "title challenge", "title contenders", "win the league",
            "top of the table", "league leaders", "pacesetters"
        ]
    )
    competitions.add_child(premier_league_competition)
    
    champions_league = TopicNode(
        name="Champions League",
        description="UEFA Champions League qualification and participation",
        keywords=[
            "champions league", "ucl", "european cup", "top four", "top 4", 
            "champions league spot", "champions league qualification", "champions league qualifier",
            "europe", "european", "european night", "champions league draw"
        ]
    )
    competitions.add_child(champions_league)
    
    europa_league = TopicNode(
        name="Europa League",
        description="UEFA Europa League qualification and participation",
        keywords=[
            "europa league", "uefa cup", "europe league", "el", "thursday night football",
            "europa league spot", "europa league qualification", "european football", 
            "europa conference league"
        ]
    )
    competitions.add_child(europa_league)
    
    fa_cup = TopicNode(
        name="FA Cup",
        description="FA Cup competition",
        keywords=[
            "fa cup", "football association cup", "fa cup tie", "fa cup draw", "fa cup run",
            "fa cup final", "fa cup semifinal", "fa cup semi-final", "fa cup quarter-final",
            "fa cup quarterfinal", "fa cup fifth round", "fa cup fourth round", "fa cup third round",
            "oldest cup competition", "wembley", "magic of the cup"
        ]
    )
    competitions.add_child(fa_cup)
    
    league_cup = TopicNode(
        name="League Cup",
        description="English League Cup (Carabao Cup, EFL Cup)",
        keywords=[
            "league cup", "carabao cup", "efl cup", "capital one cup", "coca cola cup",
            "carling cup", "worthington cup", "milk cup", "littlewoods cup", "rumbelows cup",
            "league cup final", "league cup semifinal", "league cup semi-final", "wembley"
        ]
    )
    competitions.add_child(league_cup)
    
    relegation = TopicNode(
        name="Relegation",
        description="Premier League relegation battle",
        keywords=[
            "relegation", "relegated", "drop", "go down", "bottom three", "bottom 3",
            "relegation zone", "relegation battle", "relegation fight", "survival", 
            "relegation scrap", "relegation dogfight", "trap door", "championship football",
            "stay up", "safety", "drop zone"
        ]
    )
    competitions.add_child(relegation)
    
    promotion = TopicNode(
        name="Promotion",
        description="Promotion to the Premier League",
        keywords=[
            "promotion", "promoted", "come up", "playoff", "play-off", "play off",
            "playoff final", "championship playoff", "going up", "championship promotion",
            "top flight", "premier league status", "newly promoted", "promotion race"
        ]
    )
    competitions.add_child(promotion)
    
    # Add Statistics subtopic
    statistics = TopicNode(
        name="Statistics",
        description="Premier League statistics and data",
        keywords=[
            "statistics", "stats", "data", "numbers", "figures", "record", "metric",
            "analysis", "analytics", "xg", "expected goals", "possession", "passes", 
            "shots", "tackles", "interceptions", "heat map", "touch map", "distance covered",
            "goalkeeper saves", "clean sheets", "conversion rate", "passing accuracy"
        ]
    )
    premier_league.add_child(statistics)
    
    # Add Venues subtopic
    venues = TopicNode(
        name="Venues",
        description="Premier League stadiums and venues",
        keywords=[
            "stadium", "venue", "ground", "home", "pitch", "field", "park", "arena",
            "capacity", "attendance", "sell-out", "full house", "away end", "home end",
            "stand", "terrace", "press box", "hospitality", "corporate", "home advantage",
            "fortress", "atmosphere"
        ]
    )
    premier_league.add_child(venues)
    
    # Add Finances subtopic
    finances = TopicNode(
        name="Finances",
        description="Premier League finances, revenues, and business",
        keywords=[
            "money", "finance", "financial", "revenue", "profit", "loss", "income",
            "wage", "salary", "contract", "sponsorship", "sponsor", "deal", "kit deal",
            "commercial", "broadcast", "tv money", "tv rights", "prize money", "matchday revenue",
            "ffp", "financial fair play", "balance sheet", "accounts", "takeover", "ownership"
        ]
    )
    premier_league.add_child(finances)
    
    # Add Fans subtopic
    fans = TopicNode(
        name="Fans",
        description="Premier League fans and supporters",
        keywords=[
            "fan", "supporter", "fanbase", "support", "crowd", "spectator", "away fans",
            "home fans", "loyal", "faithful", "following", "chant", "song", "banner",
            "tifo", "ultras", "atmosphere", "12th man", "season ticket", "sold out"
        ]
    )
    premier_league.add_child(fans)
    
    # Add Media subtopic
    media = TopicNode(
        name="Media",
        description="Premier League media coverage",
        keywords=[
            "media", "coverage", "broadcast", "television", "tv", "pundit", "commentator",
            "analysis", "interview", "press conference", "podcast", "sky sports", "bt sport",
            "bbc", "match of the day", "motd", "amazon prime", "stream", "highlights", "replay"
        ]
    )
    premier_league.add_child(media)
    
    # Add Refereeing subtopic
    refereeing = TopicNode(
        name="Refereeing",
        description="Premier League referees and officiating",
        keywords=[
            "referee", "ref", "official", "officiating", "var", "video assistant referee",
            "linesman", "assistant referee", "fourth official", "decision", "penalty", 
            "card", "yellow card", "red card", "sending off", "dismissed", "offside",
            "handball", "foul", "booking", "controversy", "dispute", "appeal", "overturn"
        ]
    )
    premier_league.add_child(refereeing)
    
    # Add Rules subtopic
    rules = TopicNode(
        name="Rules",
        description="Premier League and football rules",
        keywords=[
            "rule", "regulation", "law", "offside", "offside rule", "offside trap",
            "handball", "foul", "misconduct", "simulation", "diving", "time-wasting",
            "dangerous play", "serious foul play", "violent conduct", "dissent", 
            "unsporting behavior", "professional foul", "penalty", "free kick"
        ]
    )
    premier_league.add_child(rules)
    
    # Add Health subtopic
    health = TopicNode(
        name="Health",
        description="Premier League player health and injuries",
        keywords=[
            "injury", "injured", "fitness", "knock", "strain", "pull", "tear", 
            "hamstring", "muscle", "ligament", "acl", "mcl", "ankle", "foot",
            "knee", "thigh", "calf", "groin", "back", "shoulder", "concussion",
            "head injury", "medical", "scan", "recovery", "rehabilitation", "rehab",
            "return", "spell out", "layoff", "treatment", "surgery", "operation"
        ]
    )
    premier_league.add_child(health)
    
    # Add Tactics subtopic
    tactics = TopicNode(
        name="Tactics",
        description="Premier League tactics and strategies",
        keywords=[
            "tactic", "strategy", "formation", "system", "setup", "shape", "approach",
            "style", "philosophy", "press", "pressing", "high press", "counter-attack",
            "counterattack", "counter press", "gegenpressing", "possession", "defensive",
            "attacking", "build-up", "build up", "transition", "set-piece", "set piece",
            "corner", "free-kick", "penalty", "man-marking", "zonal marking", "offside trap",
            "low block", "high line", "deep line", "diamond", "4-4-2", "4-3-3", "3-5-2",
            "3-4-3", "4-2-3-1", "wing play", "overlap", "underlap", "through ball"
        ]
    )
    premier_league.add_child(tactics)
    
    return football_taxonomy


def _create_team_node(team_name: str, keywords: List[str]) -> TopicNode:
    """
    Helper function to create a team node with standard format.
    
    Args:
        team_name: Name of the team
        keywords: List of team-specific keywords
        
    Returns:
        Configured TopicNode for the team
    """
    return TopicNode(
        name=team_name,
        description=f"{team_name} football club in the English Premier League",
        keywords=keywords
    )
