{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11\n"
     ]
    }
   ],
   "source": [
    "a = 10\n",
    "print(a+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<blueSnake>"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class Snake(object):\n",
    "\tdef __init__(self, name, toxicity, aggression):\n",
    "             self.name = name\n",
    "             self.toxicity = toxicity\n",
    "             self.aggression = aggression\n",
    "\tdef __repr__(self):\n",
    "             return \"<%s>\" % self.name\n",
    "\n",
    "# Table Definitions\n",
    "gardenSnake = Snake('gardenSnake', 10, 0.1)\n",
    "gardenSnake = Snake('gardenSnake', 11, 0.424)\n",
    "bluesnake = Snake('blueSnake', 10, 0.2)\n",
    "rattleSnake = Snake('rattleSnake', 100, 0.25)\n",
    "kingCobra = Snake('kingCobra', 50, 1.0)\n",
    "snakes = [rattleSnake, kingCobra, gardenSnake, bluesnake]\n",
    "\n",
    "k = 1\n",
    "venom = 10\n",
    "strike_prob = 0.2\n",
    "\n",
    "\n",
    "\n",
    "def byDangerous_key(snake):\n",
    "    \n",
    "    c =  ((snake.toxicity - venom)**2 + (snake.aggression - strike_prob)**2)**.5\n",
    "    return c\n",
    "\n",
    "sorted_List = sorted(snakes, key = byDangerous_key)\n",
    "top_k = sorted_List[:k]\n",
    "\n",
    "from collections import Counter\n",
    "\t\n",
    "class_counts = Counter(snakes for (snakes) in top_k)\n",
    "\t\n",
    "classification = max(class_counts, key=lambda cls: class_counts[cls])\n",
    "\n",
    "classification\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
