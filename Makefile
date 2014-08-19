# .PHONY. allows to always execute the instruction, (no checking for the existence)

.PHONY.: options
options:
	@grep [a-z]: Makefile

.PHONY.: pull
pull:
	git pull origin master

.PHONY.: push
push:
	git push -u origin master
