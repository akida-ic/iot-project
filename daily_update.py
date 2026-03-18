import subprocess
import sys
from datetime import datetime

print("Pipeline started:", datetime.now())

# run Garmin fetch using current Python interpreter
subprocess.run([sys.executable, "garmin_fetch.py"])

# run Weather fetch using current Python interpreter
subprocess.run([sys.executable, "weather_fetch.py"])

print("Pipeline finished:", datetime.now())