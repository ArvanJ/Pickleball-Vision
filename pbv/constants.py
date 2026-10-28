"""Official pickleball court geometry, in metres.

Court-plane coordinates: x runs across the court (0..COURT_W), y runs along it
(0 = far baseline, COURT_L = near baseline, as seen from the camera).
"""
FT = 0.3048
COURT_W = 20 * FT        # 6.096 m
COURT_L = 44 * FT        # 13.411 m
NET_Y = COURT_L / 2
KITCHEN = 7 * FT         # non-volley zone depth on each side of the net
