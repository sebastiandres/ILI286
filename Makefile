# .PHONY. allows to always execute the instruction, (no checking for the existence)

TESTS = -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

.PHONY.: options
options:
	@grep [a-z]: Makefile

.PHONY.: pull
pull:
	git pull origin master

.PHONY.: push
push:
	git push -u origin master

.PHONY.: alltests
alltests:
	@$(foreach var, $(TESTS), echo TEST $(var); ./swept.py test_trash --test $(var);)	

.PHONY.: clean
clean:
	@find . -name "*.pyc" -delete
	@find . -name "*.py~" -delete
