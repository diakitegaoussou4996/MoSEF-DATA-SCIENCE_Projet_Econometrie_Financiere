import subprocess
import sys
import os

# # Créer un nouvel environnement virtuel
# subprocess.check_call([sys.executable, '-m', 'venv', 'venv_st'])

# # Activer l'environnement virtuel
# activate_this = os.path.join('venv_st', 'Scripts', 'activate')
# # with open(activate_this) as file:
# #     exec(file.read(), {'__file__': activate_this})


# subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])

import script  

if __name__ == '__main__':
    script.main()  
