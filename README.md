**IMPORTANT**: Follow the instructions below.

# Hecktor 2022 Code

All code from members in the Qurit group working on the Hecktor 2022 challenge. This is a private repository.

# Instructions

Please add your code in the following way.

1. Pull the repository to your computer using `git pull git@github.com:qurit/hecktor_2022.git`. In order to do this, you will need to generate an ssh key on the computer you're working on and upload it to Github. For a guide on how to do that, see here https://docs.github.com/en/authentication/connecting-to-github-with-ssh. This is required since this repository is private and only we have access to it.
2. On the comptuer that you're working on, you should have a folder that corresponds to the Git repository. Once inside that folder, create your own branch for development of your own code. For example, for Luke Polson, I would do `git checkout -b lpolson`
3. Create your own folder for your own code. In the main folder of the repository, I would do `mkdir lpolson`. All your own code should be added to this folder; you can structure things however you like.
4. Most importantly *when pushing to Github* (to share your code) **only push to your development branch**. For example, for myself, I would always do `git checkout lpolson` (you should be working on your code only in your own branch anyways) and then `git push`. In fact, the rules are set up so that you won't be allowed to push to main.
5. If you ever want to merge to the `main` branch, submit a pull request through Github. This can be found on the "pull requests" tab above.
6. Since you will be working in your own environment **create a docker image inside your folder** that includes all packages one needs to run your code. When you're working on your own code, develop it in Docker.

