

SCENE?=13

.PHONY: train 
train:
	python3 ./ppo.py --train -s ${SCENE}


.PHONY: train-cont
train-cont:
	python3 ./ppo.py --train --resume -s ${SCENE}


.PHONY: test
test:
	python3 ./ppo.py --render -s ${SCENE}



.PHONY: render
render:
	python3 ./ppo.py --train --render -s ${SCENE}


.PHONE: render-cont
render-cont:
	python3 ./ppo.py --train --render --resume -s ${SCENE}
