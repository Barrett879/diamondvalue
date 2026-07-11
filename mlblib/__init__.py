"""DiamondValue: MLB per-game player projection site.

Package layout:
  cache.py     disk-cache toolkit (ported from HoopsValue) + date-keyed freshness
  fetch.py     MLB Stats API + Baseball Savant, all stale-beats-empty
  features.py  point-in-time feature construction, shared by training + inference
  model.py     artifact loading, feature-vector assembly, prediction
  theme.py     token-based light/dark theme, chrome, nav, footer, SENTINEL
"""
