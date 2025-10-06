from setuptools import setup, find_packages

# get_requirements() = helper function that parses requirements.txt into a clean Python list for setup.py.
def get_requirements(file_path:str)->list[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(    
    name='mlproject',
    version='0.0.1',
    author='Mounika Maradana',
    author_email='maradana.mounika17@gmail.com',
    packages=find_packages(), # find_packages for all the folders where we have the __init__.py file and treat all those folders as packages
    install_requires=get_requirements('requirements.txt') # install_requires is an argument to setup() and tell Python which dependencies your package needs to work. These dependencies will be installed automatically when someone installs your package with pip.
    )
