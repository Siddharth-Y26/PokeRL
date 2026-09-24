"""Live coaching: explain a trained policy's choice to a human mid-battle.

The study's agents are evaluated by win rate alone. This package turns one of them into a
teacher — given a battle position, it reports what it would play, how strongly it prefers
that, and which parts of the position drive the preference.

``reconstruct``  rebuilds a poke-env ``Battle`` from Showdown protocol traffic forwarded by
                 the browser client, so the coach sees exactly what the human sees
``explain``      turns a battle position into a recommendation plus its reasons
"""
