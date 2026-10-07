import json
import pytest
from agents import niche, us_trends, researcher, topic_validation


def verdict(domain='MOVIE', **updates):
    result = dict(candidate_id='x', content_domain=domain, content_type='EXPLAINER',
                  niche_subject='Avengers', niche_evidence='A new Avengers movie trailer',
                  us_movie_relevant=True, us_relevance_reason='Hollywood production', confidence=0.95)
    result.update(updates)
    return result


@pytest.mark.parametrize('title,domain', [
    ('Avengers movie trailer', 'MOVIE'),
    ('Upcoming Jurassic World release announcement', 'MOVIE'),
    ('Inception ending explained', 'MOVIE'),
    ('Dragon Ball transformation explained', 'ANIME'),
    ('One Piece new season announcement', 'ANIME'),
    ('Upcoming Demon Slayer anime movie', 'ANIME'),
    ('International anime adaptation announcement', 'ANIME'),
])
def test_niche_classifications(monkeypatch, title, domain):
    response = verdict(domain, niche_subject=title, niche_evidence=title,
                       us_movie_relevant=domain == 'MOVIE',
                       us_relevance_reason='Hollywood production' if domain == 'MOVIE' else '')
    monkeypatch.setattr(niche, 'generate', lambda *a, **k: json.dumps(response))
    selected = niche.select_semantic_candidate([dict(video_id='x', title=title)])
    assert selected['content_domain'] == domain


@pytest.mark.parametrize('title,domain', [
    ('US viral challenge', 'GENERAL_TREND'), ('Election debate', 'POLITICS'),
    ('NBA Finals recap', 'SPORTS'), ('AI chip launch', 'TECHNOLOGY'),
    ('Stock market update', 'BUSINESS'), ('Celebrity dating gossip', 'GOSSIP'),
    ('Unrelated cooking tips', 'OTHER'),
])
def test_out_of_niche_verdicts_rejected(monkeypatch, title, domain):
    monkeypatch.setattr(niche, 'generate', lambda *a, **k: json.dumps(verdict(domain)))
    with pytest.raises(RuntimeError, match='niche restriction'):
        niche.select_semantic_candidate([dict(video_id='x', title=title)])


@pytest.mark.parametrize('updates', [
    dict(us_movie_relevant=False), dict(us_movie_relevant='true'),
    dict(us_movie_relevant=None), dict(us_relevance_reason=''),
    dict(niche_subject=''), dict(niche_evidence=''),
    dict(content_domain=''), dict(niche_evidence=None),
])
def test_ambiguous_or_non_us_movie_rejected(monkeypatch, updates):
    monkeypatch.setattr(niche, 'generate', lambda *a, **k: json.dumps(verdict(**updates)))
    with pytest.raises(RuntimeError):
        niche.select_semantic_candidate([dict(video_id='x', title='Movie announcement')])


def test_rejected_batch_continues_to_anime(monkeypatch):
    monkeypatch.setattr(us_trends.config, 'DISCOVERY_SEMANTIC_BATCH_SIZE', 1)
    replies = iter([verdict('SPORTS'), verdict('ANIME', candidate_id='anime', us_movie_relevant=False)])
    monkeypatch.setattr(niche, 'generate', lambda *a, **k: json.dumps(next(replies)))
    selected = us_trends.choose_best_trend([
        dict(video_id='x', title='NBA Finals', trend_score=100),
        dict(video_id='anime', title='Dragon Ball transformation', trend_score=90),
    ])
    assert selected['content_domain'] == 'ANIME'


def test_movie_duplicate_protection_unchanged(monkeypatch):
    monkeypatch.setattr(us_trends, 'load_used_topics', lambda: ['Avengers movie trailer'])
    assert us_trends.filter_candidates([dict(video_id='x', title='Avengers movie trailer explained')]) == []


def test_research_cannot_change_domain(monkeypatch):
    monkeypatch.setattr(researcher, 'generate', lambda *a, **k: json.dumps(dict(
        topic='Avengers movie', source_trend='Avengers movie', video_title='Avengers explained',
        content_domain='GENERAL_TREND', content_type='EXPLAINER', key_points=['Avengers'], tags=[])))
    with pytest.raises(ValueError, match='domain drifted'):
        researcher._research_selected_candidate(dict(selected_trend='Avengers movie', content_domain='MOVIE'), '', [])


def test_anime_research_domain_allowed():
    topic_validation.validate_research_topic_lock(dict(
        topic='Dragon Ball transformation', source_trend='Dragon Ball transformation',
        content_domain='ANIME', content_type='EXPLAINER'))
