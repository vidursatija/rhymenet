import numpy as np
import collections
import os
import nltk
import pickle

class LyricsReader():

	def __init__(self, start_year, end_year, folder_path, vocab_size):
		self.num_years = end_year-start_year+1
		self.start_year = start_year
		self.path = folder_path
		self.vocab_size = vocab_size
		self.arpabet = nltk.corpus.cmudict.dict()

		all_words = []
		all_phenomes = []
		all_lyrics = []
		self.all_lyrics_path = []
		#all_lyrics.append([])
		for i in range(self.num_years):
			year_path = str(folder_path+"/"+str(i+start_year))
			print(str(i+self.start_year))
			for file in os.listdir(year_path):
				if file.endswith(".300") == False:
					all_lyrics.append([])
					song_path = os.path.join(year_path, file)
					self.all_lyrics_path.append(song_path)
					#print(song_path)
					with open(song_path, 'r') as f:
						#all_lyrics[-1] = []
						for each_line in f:
							words = nltk.word_tokenize(each_line)
							add_word = True
							for word in words:
								if word == '[' or word == '(':
									add_word = False
								if word == ']' or word == ')':
									add_word = True
									continue
								if add_word == False:
									continue
								if word[0].isalpha() == False:
									continue
								rt = 'UNKR'
								try:
									rt = self.arpabet[word.lower()][0][-1]
								except Exception as e:
									pass
								if rt[-1] == '0' or rt[-1] == '1' or rt[-1] == '2':
									rt = rt[:-1]
								all_phenomes.append(rt)
								all_words.append(word.lower())
								all_lyrics[-1].append(word.lower())
							if each_line != '\n':
								all_words.append('new_line')
								all_phenomes.append('NEWR')
								all_lyrics[-1].append('new_line')
								#continue
							"""if len(words) == 0:
								if len(all_lyrics[-1]) < 7:
									all_lyrics[-1] = []
								else:
									all_lyrics.append([])"""

		self.count = len(all_lyrics)
		print(len(all_lyrics))

		counter = collections.Counter(all_words)
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
		count_pairs = count_pairs[:self.vocab_size-2]
		words, _ = list(zip(*count_pairs))
		self.word_to_id = dict(zip(words, range(len(words))))
		self.word_to_id['UNKT'] = vocab_size-1
		self.word_to_id['FINT'] = vocab_size-2

		counter = collections.Counter(all_phenomes)
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
		phenomes, _ = list(zip(*count_pairs))
		self.phenome_to_id = dict(zip(phenomes, range(len(phenomes))))
		self.phenome_to_id['FINR'] = len(phenomes)
		print(len(self.phenome_to_id))

		self.word_id_to_phenome_id = [0] * vocab_size
		for key, value in self.word_to_id.items():
			if key == 'UNKT':
				self.word_id_to_phenome_id[self.word_to_id[key]] = self.phenome_to_id['UNKR']
				continue
			if key == 'FINT':
				self.word_id_to_phenome_id[self.word_to_id[key]] = self.phenome_to_id['FINR']
				continue
			if key == 'new_line':
				self.word_id_to_phenome_id[self.word_to_id[key]] = self.phenome_to_id['NEWR']
				continue
			try:
				self.word_id_to_phenome_id[self.word_to_id[key]] = self.phenome_to_id[self.arpabet[key.lower()][0][-1]]
			except:
				self.word_id_to_phenome_id[self.word_to_id[key]] = self.phenome_to_id['UNKR']

		#print(self.phenome_to_id)
		#print(len(self.word_to_id))
		#print(len(self.phenome_to_id))

		self.all_lyrics = []
		self.all_rhymes = []
		for i in range(self.count):
			self.all_lyrics.append([])
			self.all_rhymes.append([])
			self.all_lyrics[i] = []
			self.all_rhymes[i] = []
			for each_word in all_lyrics[i]:
				if each_word == 'new_line':
					self.all_lyrics[i].append(self.word_to_id['new_line'])
					self.all_rhymes[i].append(self.phenome_to_id['NEWR'])
					continue

				try:
					self.all_lyrics[i].append(self.word_to_id[each_word.lower()])
				except Exception as e:
					self.all_lyrics[i].append(vocab_size-1)

				try:
					rt = self.arpabet[each_word.lower()][0][-1]
					if rt[-1] == '0' or rt[-1] == '1' or rt[-1] == '2':
						rt = rt[:-1]
					self.all_rhymes[i].append(self.phenome_to_id[rt])
				except Exception as e:
					#print(e)
					self.all_rhymes[i].append(self.phenome_to_id['UNKR'])

			self.all_lyrics[i] = self.all_lyrics[i][1:-1] + [self.word_to_id['FINT']]
			self.all_rhymes[i] = self.all_rhymes[i][1:-1] + [self.phenome_to_id['FINR']]

		#print(self.all_lyrics[2])
		#print(self.all_rhymes[2])
		#print(self.phenome_to_id)
		#print(np.array(self.all_lyrics[2]).shape)

	def getSongsCount(self):
		return self.count

	def lyricsAtIndex(self, i):
		return self.all_lyrics[i]

	def rhymesAtIndex(self, i):
		return self.all_rhymes[i]

	def pathAtIndex(self, i):
		return self.all_lyrics_path[i]

	

if __name__ == '__main__':
	l = LyricsReader(2008, 2016, 'lyrics_scraper/r-b-hip-hop-songs', 10000)
	with open('lyrics_scraper/rap10k.pickle', 'wb') as f:
		pickle.dump({'count': l.count, 'all_lyrics': l.all_lyrics, 'all_rhymes': l.all_rhymes, 'w2id': l.word_to_id, 'p2id': l.phenome_to_id, 'wi2pi': l.word_id_to_phenome_id}, f, protocol=2)
		#pass
	print("Test")
