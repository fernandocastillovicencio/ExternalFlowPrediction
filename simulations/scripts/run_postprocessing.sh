# !/bin/bash



# ---------------- CREATE CLOUD DICTIONARY --------------- #
# Path to the postprocessing template directory
TEMPLATE_DIR="../template/postprocessing"

# Check if the template directory exists
[ -d "$TEMPLATE_DIR" ] || { echo "The directory $TEMPLATE_DIR does not exist!" && exit 1; }

# Execute generateCloud script
python3 generate_cloudDict.py

# Copy postprocessing files to each case directory
find ../cases -type d -name 'Re_*' -exec cp -rp "$TEMPLATE_DIR"/* {} \;

# Execute postProcess commands in parallel
find ../cases -type d -name 'Re_*' | parallel -j $(nproc) "cd {} && postProcess -func cloud -latestTime"


# --------------- FIX MISSING CLOUD POINTS --------------- #
# Defina o diretório das simulações
SIMULATIONS_DIR=../cases
# Defina o comando para corrigir os pontos
CORRECT_POINTS_CMD="python fix_missing_cloud_points.py"
# Use o comando parallel para executar o comando em paralelo em cada pasta
parallel "$CORRECT_POINTS_CMD {}" ::: $(find "$SIMULATIONS_DIR" -type d -name "Re_*")