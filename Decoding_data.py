import json

# ptbk responds
ptbk_resp = {"status":0,"data":"i9B3kbt7sNSg%Y013208476+9,5-%."}
ptbk = ptbk_resp["data"]

# Encrypting data
enc_data = "BkNS9b7S9g3SB77SBktS9b9S9tBS9N3Sb3gS9t7S9k7Sb3iS993SBNNS93gS93BS99iSBNbSBktS9i7S99bS9B9S9BiS99kS9i3SBN3S999S9bBS9BbSb3iS973S9BiS9BbS9BbS9BNS93bS9BiS99NS9BtSBttSBNgS9i9S93kS93NSBk7S93tS9B7S9igS93bS9ibS93kSBttSBbNSBN3S937SBNtS93NS9ibSBb9S9iBS9gNS9N3S9kkS9NiS9gtS9B9S93bS9giS973S9B7S9ikS99iS9tbS9BbS9BbS9i9S9BbSbkgSbitS99BS99bS99iSBktS93tS939S9BtSBNBS93BS999S939SBtNSBk9SBtbSBNkS9iNSBNiSBNgS93tS999S9g7S9btS9ikS99NSBkiSB73Sb33S9BgS9BNSBktSB9BS973S9gtSbiBS9k7S97gSbi7S9N9S9N7S9k3S9k3S9tbS9gBS9bBS979S9bBS979S9BNS9k7Sg9tSg3BS7igSb33S977Sb3NSbiiS9bkS9ktS977S9N9S9kbS97NS9kkSbBbS9tiS9g7SbiBS9kkSb3BS9k7S9tNSb37SbBbS9kgS9N3S9b3S997SBN7S937S9B9SBk9SBNNS97bS9tiSbBgSbibSbgtSbibSb99SbbbSbikSbiiS9kNSbiiS9kBS979S9k9S9NkS9tkS9ttS9N9Sb3gSb9kS7bBSb3tSb37SbikSbigS9NBS97NS97bS9bNSktkSgbBSbgiSt3bSbkNSgikSgNgSgBiSbBiS9NbS9NtSbbiSb93S9k7S9ggS9BgS9bBS937SBNtS9B3S9i3S9ibSBN7SBt3SBkiS9gtSb9tS9tiS9ktS97gS99bS9B9S97bS9b9S9BbSBtNS9ggS9bNS99kS9BNS9gbS933S99iSbttSb3NS99kS9gNS9gkS999SBNNS9i3S93tS9BkS9btS9ggSBNBS9bgSBgiS9g9S993S9b9S9biS93bS93bS9i9S9i9SBNtS939SBNtSBk3SBgtSBt3SBggSB97SiNBSBbkSBgbSBtkSBk9SBN3SBkbSBkNSBN3SBg9SBNBSBbkSBgNSBitSBkBSBk9S9i3S93NSBtNSBt3S93tS9N3S9BbSBt3SBgkSB77SBgiSBbbSB7BSB9NSBgNSBb3SBk7SB79"  # 你粘贴的很长那串

# Creating a mapping table
half = len(ptbk) // 2
mapping = {ptbk[i]: ptbk[i+half] for i in range(half)}

# decoding
decoded = "".join(mapping.get(ch, ch) for ch in enc_data)

# Split by comma
values = decoded.split(",")

# Generating CSV
import pandas as pd
from datetime import datetime, timedelta

start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
dates = [start_date + timedelta(days=7*i) for i in range(len(values))]

df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                   "value": values})

print(df.head(299))
df.to_csv("baidu_index.csv", index=False)
