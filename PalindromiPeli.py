
# Copyright (c) 2024 Jari Hiltunen / GitHub Divergentti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Finnish verbs, adjectives and substantives CC-BY by Kotimaisten kielten keskus


import csv
import re
import os
import sys
import string
import asyncio
import textwrap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QDialog, QMenu)
from PyQt6 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from main_form import Ui_first_window
from nltk_form import Ui_NLTKDialog
from graph_visualization import Ui_graph_Dialog
from generator import Ui_generate_palindromes_Dialog
from inspect_palindromes import Ui_inspect_Dialog
from game_instructions import Ui_game_instructions_Dialog
from qasync import QEventLoop, asyncSlot
import json
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
import nltk
import logging
import pandas as pd

try:
    with open('runtimeconfig.json', 'r') as config_file:
        data = json.load(config_file)
    verbs_file = data.get('verbs_file')
    adjectives_file = data.get('adjectives_file')
    substantives_file = data.get('substantives_file')
    long_sentences_file = data.get('long_text_file')
    new_palindromes_file = data.get('new_palindromes_file')
    converted_palindromes_file = data.get('converted_palindromes_file')
    new_subs_palindromes_file = data['new_subs_palindromes_file']
    new_verb_palindromes_file = data['new_verb_palindromes_file']
    new_adj_palindromes_file = data['new_adj_palindromes_file']
    new_long_text_palindromes_file = data['new_long_text_palindromes_file']

except OSError as err:
    print("Error with runtimeconfig.json: ", err)
except json.JSONDecodeError as json_err:
    print("Error decoding JSON: ", json_err)

# a few globals, mainly for PalindromeMaker
begin_word =""
fail_counter = 0

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class FEEDER(object):
    """
       Feeder load csv- and text-files, clean them, remove duplicates etc. Main class for other classes!
       You need to confifure filenams in runtimeconfig.json.
       Filenames beginning with new_ are used in GENERATE-class for new palindromes generation.

       For degugging, set "debug=True" during object initialization

    """
    # ANSI escape codes for colors
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BLUE = "\033[94m"
    COLOR_RED = "\033[91m"

    def __init__(self, debug=False):
        """
               Constructor:
               Input (from runtimeconfig.json): verb, adjectives, substantives, long sentences (from book etc),
               and filename for new palindromes.

               Args: debug
               """
        self.adj_anagrams = None
        self.subs_anagrams = None
        self.verb_anagrams = None
        self.sentences = None
        self.debug = debug
        self.clean_long_sentences = None
        self.clean_substantives = None
        self.clean_adjectives = None
        self.clean_verbs = None
        self.extracted_words = []  # from long text such as a book
        self.verb_anagrams = []  # anagrams = mirror words such as ISI
        self.subs_anagrams = []
        self.adj_anagrams = []
        self.long_anagrams = []
        self.new_palindromes = []
        self.failed_tries = []

        if verbs_file and os.path.exists(verbs_file):
            self.verbs = self.load_words(verbs_file)
            self.clean_verbs = set(self.remove_duplicates(self.verbs))  # remove duplicates
            self.word_anagrams_in_lists(self.clean_verbs, self.verb_anagrams)  # find anagramic words
            if self.debug:
                print(f"{self.COLOR_GREEN}Clean verbs loaded: {self.clean_verbs}{self.COLOR_RESET}")
                print(f"{self.COLOR_GREEN}Verb anagrams: {self.verb_anagrams}{self.COLOR_RESET}")

        if adjectives_file and os.path.exists(adjectives_file):
            self.adjectives = self.load_words(adjectives_file)
            self.clean_adjectives = set(self.remove_duplicates(self.adjectives))
            self.word_anagrams_in_lists(self.clean_adjectives, self.adj_anagrams)
            if self.debug:
                print(f"{self.COLOR_YELLOW}Clean adjectives loaded: {self.clean_adjectives}{self.COLOR_RESET}")
                print(f"{self.COLOR_YELLOW}Adjective anagrams: {self.adj_anagrams}{self.COLOR_RESET}")

        if substantives_file and os.path.exists(substantives_file):
            self.substantives = self.load_words(substantives_file)
            self.clean_substantives =set(self.remove_duplicates(self.substantives))
            self.word_anagrams_in_lists(self.clean_substantives, self.subs_anagrams)
            if self.debug:
                print(f"{self.COLOR_BLUE}Clean substantives loaded: {self.clean_substantives}{self.COLOR_RESET}")
                print(f"{self.COLOR_BLUE}Substantive anagrams: {self.subs_anagrams}{self.COLOR_RESET}")

        if long_sentences_file and os.path.exists(long_sentences_file):
            """ In Finnish language we have complex syntax. Use book etc for more complex words"""
            self.long_sentences = self.load_sentences(long_sentences_file)  # note! txt-file!
            self.clean_long_sentences = set(self.remove_duplicates(self.long_sentences))
            self.extracted_words = self.remove_duplicates(self.extract_words_from_sentences(self.clean_long_sentences))
            filtered_words = [word for word in self.extracted_words if len(word) >= 2]  # filter words less than 2 chars
            self.extracted_words = filtered_words
            adjectives_set = set(self.clean_adjectives) if adjectives_file else set()
            verbs_set = set(self.clean_verbs) if verbs_file else set()
            substantives_set = set(self.clean_substantives) if substantives_file else set()
            self.extracted_words = [word for word in self.extracted_words
                                    if word not in adjectives_set and word not in verbs_set and word not in substantives_set]
            self.word_anagrams_in_lists(self.extracted_words, self.long_anagrams)
            if self.debug:
                print(f"{self.COLOR_RED}Words from long sentences after cleaning: {self.extracted_words}{self.COLOR_RESET}")


    def check_palindromes(self, word_list):
        """ Verify if palindrome = anagram too """
        return [word for word in word_list if word == word[::-1]]


    def load_words(self, file_name):
        """ Reads csv-file containing words, comma separated """
        try:
            with open(file_name, newline='') as f:
                reader = csv.reader(f)
                self.extracted_words = [row[0] for row in reader if row]
            return self.extracted_words
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def load_text_rows(self, file_name):
        """ Reads csv-file containing words, comma separated """
        try:
            with open(file_name, "r") as file:  # comma separated values
                text_rows = list(csv.reader(file, delimiter=","))
            text_rows = [item for sublist in text_rows for item in sublist if item]
            return text_rows
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def load_sentences(self,file_name):
        """ Reads txt-file containing lines """
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                self.sentences = f.readlines()
            self.sentences = [line.strip() for line in self.sentences if line.strip()]
            return self.sentences
        except Exception as e:
            logger.error("Error: %s", e)
            if self.debug:
                print("Error: %s", e)

    def extract_words_from_sentences(self, sentences):
        """ Strip words our from lines """
        words_out = []
        for sentence in sentences:
            words_in_sentence = sentence.split()
            cleaned_words = [self.clean_text(word) for word in words_in_sentence]
            words_out.extend(cleaned_words)
        return words_out

    @staticmethod
    def clean_text(text):
        """ Clean text, leave only alphabets """
        text = re.sub(r'[^a-zA-ZäöåÄÖÅ]', '', text)
        return text.lower()

    @staticmethod
    def remove_duplicates(input_list):
        """ Remove duplicate entries from lists """
        seen = set()
        unique_list = []
        for item in input_list:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)
        return unique_list

    def remove_duplicates_with_spaces(self, palindrome_list):
        """ Remove duplicates from a list, preserving spaces and keeping the original order """
        seen = set()  # To track seen palindromes
        unique_palindromes = []  # List to store unique palindromes

        for palindrome in palindrome_list:
            cleaned_palindrome = palindrome.lower().strip()  # Clean spaces and make lowercase
            if cleaned_palindrome not in seen:
                unique_palindromes.append(palindrome)  # Add original (with spaces)
                seen.add(cleaned_palindrome)  # Track cleaned version to avoid duplicates

        return unique_palindromes

    def save_new_palindromes(self, palindromes, file_name):
        """ Save new palindromes to the file, avoiding duplicates
            This information could be used for learning ML algorithms
        """

        # Load existing palindromes from the file
        existing_palindromes = set()  # Using a set to avoid duplicates
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if row:
                            existing_palindromes.add(row[0].strip())  # Add existing palindromes to the set
            except Exception as e:
                logger.error("Error: %s", e)
                if self.debug:
                    print("Error: %s", e)

        # Remove duplicates in the new palindromes (and check against existing palindromes)
        new_unique_palindromes = self.remove_duplicates_with_spaces(palindromes)
        new_palindromes_to_save = [p for p in new_unique_palindromes if p.strip() not in existing_palindromes]

        # Append the new, unique palindromes to the file
        if new_palindromes_to_save:
            try:
                with open(file_name, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for palindrome in new_palindromes_to_save:
                        writer.writerow([palindrome])
            except Exception as e:
                logger.error("Error: %s", e)
                if self.debug:
                    print("Error: %s", e)
            if self.debug:
                print(f"Saved {len(new_palindromes_to_save)} new palindromes to {file_name}")

    def add_failed_try(self, word):
        """ Add non-palindromic word (fail) to the list """
        self.failed_tries.append(word)

    def word_anagrams_in_lists(self, word_list, anagram_list):
        """ Generalized palindrome check for word lists """
        for word in word_list:
            if word == word[::-1]:
                anagram_list.append(word)
                if self.debug:
                    print(f"Anagram found: {word}")


class PalindromeMaker:
    """ This class uses words loaded from the Feeder and then use symmetric logics to make new palindromes
           for ML-learning and for the game.

        Constructor: initialized by GENERATOR form class, passing variables (super)

        Args: debug, status (for screen updates in GENERATOR form), chosen_wordlist (from GENERATOR),
              new_file (based on user selection at GENERATOR form), cancel_requested (controlled from GENERATOR)

    """

    def __init__(self, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.status = "Not running"
        self.chosen_wordlist = None  # set from GENERATE class
        self.new_file = None  # new_adj, verb, subs, long.csv set from GENERATE class
        self.cancel_requested = False  # interrupt handler

    def is_anagram(self, text):
        """ Check if anagram (mirror) """
        return text == text[::-1]

    def make_symmetric(self, text):
        """ Make symmetric = mirror"""
        return text + text[::-1]

    def iterate_alphabet_characters(self, word, position):
        """ First level iterator: begin with adding character into middle of the mirrored (anagram) word """
        global fail_counter
        found_palindrome = False
        first_letter = ""
        palindrome = ""
        finnish_alphabet = string.ascii_lowercase + 'äöå'
        index = len(word) + position
        for letter in finnish_alphabet:
            if self.cancel_requested:
                if self.debug:
                    print("Generation cancelled!")
                return
            new_word = word[:index] + letter + word[index:] + word[::-1]
            if self.make_sense(new_word):
                if new_word == new_word[::-1]:
                    found_palindrome = True
                    first_letter = letter
                    palindrome = new_word
        if found_palindrome is True:
            feed.new_palindromes.append(palindrome)
            used_words = set()
            self.extend_palindrome_second_phase(palindrome, first_letter, index, used_words)
        else:
            fail_counter += 1
            return False

    def find_palindrome_extensions_first_letter(self, first_letter, used_words):
        """ Return verbs etc. based on first letter of the word or sentence """
        ext_verbs = [w for w in feed.clean_verbs if w.lower().startswith(first_letter) and w not in used_words]
        ext_adjectives = [w for w in feed.clean_adjectives if
                          w.lower().startswith(first_letter) and w not in used_words]
        ext_substantives = [w for w in feed.clean_substantives if
                            w.lower().startswith(first_letter) and w not in used_words]
        ext_txt_words = [w for w in feed.extracted_words if w.lower().startswith(first_letter) and w not in used_words]
        return ext_verbs + ext_adjectives + ext_substantives + ext_txt_words

    def extend_palindrome_second_phase(self, palindrome, first_letter, index, used_words):
        """ Second phase iterator continue expanding the palindrome by inserting words beginning with
        the same character as in the middle """
        extensions = self.find_palindrome_extensions_first_letter(first_letter, used_words)
        for ext_word in extensions:
            extended_palindrome = ' '.join([palindrome[:index], ext_word, palindrome[index:]])
            if extended_palindrome.replace(' ', '') == extended_palindrome.replace(' ', '')[::-1]:
                feed.new_palindromes.append(extended_palindrome)
                used_words.add(ext_word)
                self.extend_palindrome_second_phase(extended_palindrome.replace(' ', ''), first_letter,
                                               index + len(ext_word), used_words)
            else:
                # Add failed tries (words) to the list
                # This could be also saved, but is big file, easily a few hundred GIGABYTES
                feed.add_failed_try(ext_word)

    def make_sense(self, anagram):
        """ Test if word makes sense = is found from vocabulary based on FEEDER words """
        first_letter = anagram[0].lower()
        first_match = False

        # Check first part
        matching_verbs = [w for w in feed.clean_verbs if w.lower().startswith(first_letter)]
        matching_adjectives = [w for w in feed.clean_adjectives if w.lower().startswith(first_letter)]
        matching_substantives = [w for w in feed.clean_substantives if w.lower().startswith(first_letter)]
        matching_txt_words = [w for w in feed.extracted_words if w.lower().startswith(first_letter)]

        # Check if word is in the lists
        if (begin_word in matching_verbs or begin_word in matching_adjectives or begin_word in matching_substantives
                or begin_word in matching_txt_words):
            first_match = True

        # If the word is found, continue to next word
        if first_match:
            remaining_part = anagram[len(begin_word):]  # rest part
            if self.check_remaining_part_second_phase(remaining_part) is True:
                return True
        else:
            return False

    def convert_new_csv_to_json(self):
        # Files which mush exist
        required_files = [
            new_subs_palindromes_file,
            new_verb_palindromes_file,
            new_adj_palindromes_file,
            new_long_text_palindromes_file
        ]

        # Check existence
        missing_files = [file for file in required_files if not os.path.exists(file)]

        if missing_files:
            self.status = f"Missing files: {', '.join(missing_files)}. Need all four before proceeding!"
            if self.debug:
                print(f"Missing files: {', '.join(missing_files)}. Need all four before proceeding!")
            return False

        # Load csv file prior to conversion
        subs_palindromes = None
        verb_palindromes = None
        adj_palindromes = None
        long_text_palindromes = None

        try:
            subs_palindromes = pd.read_csv(new_subs_palindromes_file, header=None)
        except Exception as e:
            self.status = ("Error reading %s: %s", new_subs_palindromes_file, e)
            logger.error("Error reading %s: %s", new_subs_palindromes_file, e)
            if self.debug:
                print("Error: %s", e)

        try:
            verb_palindromes = pd.read_csv(new_verb_palindromes_file, header=None)
        except Exception as e:
            self.status = ("Error reading %s: %s", new_verb_palindromes_file, e)
            logger.error("Error reading %s: %s", new_verb_palindromes_file, e)
            if self.debug:
                print("Error: %s", e)

        try:
            adj_palindromes = pd.read_csv(new_adj_palindromes_file, header=None)
        except Exception as e:
            self.status = ("Error reading %s: %s", new_adj_palindromes_file, e)
            logger.error("Error reading %s: %s", new_adj_palindromes_file, e)
            if self.debug:
                print("Error: %s", e)

        try:
            long_text_palindromes = pd.read_csv(new_long_text_palindromes_file, header=None)
        except Exception as e:
            self.status =("Error reading %s: %s", new_long_text_palindromes_file, e)
            logger.error("Error reading %s: %s", new_long_text_palindromes_file, e)
            if self.debug:
                print("Error: %s", e)

        # Check that all files are loaded correctly
        if any(df is None for df in [subs_palindromes, verb_palindromes, adj_palindromes, long_text_palindromes]):
            self.status = "One or more files failed to load."
            if self.debug:
                print("One or more files failed to load.")
            return False

        # Combine dataframes
        dataframes = [df for df in [subs_palindromes, verb_palindromes, adj_palindromes, long_text_palindromes] if
                      df is not None]
        combined_palindromes = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates
        combined_palindromes.drop_duplicates(subset=[0], inplace=True)
        palindromes_list = combined_palindromes[0].tolist()  # convert to lists

        # Save to json
        try:
            with open(converted_palindromes_file, 'w', encoding='utf-8') as f:
                json.dump(palindromes_list, f, ensure_ascii=False, indent=4)
            # If successful, set status message here
            self.status = "CSV to JSON conversion complete and saved!"
        except Exception as e:
            self.status = f"Error saving to {converted_palindromes_file}: {e}"
            logger.error("Error saving to %s: %s", converted_palindromes_file, e)
            if self.debug:
                print("Error: %s", e)

    def check_remaining_part_second_phase(self,remaining_part):
        """ If there is more to check about the palindrome"""
        if remaining_part:
            first_letter = remaining_part[0].lower()

            # Check rest part in vocabulary
            matching_verbs = [w for w in feed.clean_verbs if w.lower().startswith(first_letter)]
            matching_adjectives = [w for w in feed.clean_adjectives if w.lower().startswith(first_letter)]
            matching_substantives = [w for w in feed.clean_substantives if w.lower().startswith(first_letter)]
            matching_txt_words = [w for w in feed.extracted_words if w.lower().startswith(first_letter)]

            # If found, return true
            if (remaining_part in matching_verbs or remaining_part in matching_adjectives
                    or remaining_part in matching_substantives or remaining_part in matching_txt_words):
                return True

    def cancel_generation(self):
        """ Set interrupt handler """
        self.cancel_requested = True

    def save_progress(self):
        """ Save the current progress of new palindromes - control from GENERATOR class!"""
        feed.save_new_palindromes(feed.new_palindromes, self.new_file)
        self.status = "Progress saved!"
        if self.debug:
            print(f"Progress saved!")

    async def make_palindromes_for_learning(self, max_palindromes=200000):
        """
        Create for ML algorithm some learning data and palindromes for the game.
        This example tries each Finnish verb (or other word list) and if a palindrome is found,
        it will store it to new_palindromes. This will run until the max_amount is reached.

        From Finnish verbs, adjectives etc. already 21000 + palindromes generated. See palindromes.json!

        - max_palindromes: making sure generations stops some day.
        """
        global feed
        global begin_word
        iteration_count = 0  # Counter for positive (found) palindromes

        # !! Main loop for generating palindromes !!

        if self.chosen_wordlist and not self.cancel_requested:
            # First level iterator produces 1-3 words
            for begin_word in self.chosen_wordlist:
                if self.cancel_requested:
                    if self.debug:
                        print("Generation interrupted! (chosen_wordlist)")
                    return

                if iteration_count >= max_palindromes:
                    # self.status is updated to user form GENERATOR
                    self.status = f"Reached maximum palindromes: {max_palindromes}"
                    logging.error(f"Reached maximum palindromes: {max_palindromes}")
                    if self.debug:
                        print(f"Reached maximum palindromes: {max_palindromes}")
                    break

                if self.iterate_alphabet_characters(begin_word, 0):
                    iteration_count += 1  # Negative result (failed) palindrome

                await asyncio.sleep(0)

            # Final save after all iterations, unless interrupted
            if not self.cancel_requested:
                self.save_progress()

            await asyncio.sleep(0)
        else:
            if self.debug:
                print("Wordlist for Generator - make_palindromes_for_learning empty!")

    async def next_level_iterator(self):
        """ If first and/or second level palindromes are found, tries to add more text in between """
        previous_length = len(feed.new_palindromes)

        while True:
            # Check if the palindrome list has grown
            if len(feed.new_palindromes) > previous_length:
                self.status = "New palindrome added! Adding anagram word to both ends..."
                if self.debug:
                    print("New palindrome added! Adding anagram word to both ends...")
                new_palindrome = feed.new_palindromes[-1]  # Get the latest palindrome

                # Extend the palindrome with anagrams
                await self.extend_palindrome_next_level(new_palindrome)

                # Update the previous length
                previous_length = len(feed.new_palindromes)

            await asyncio.sleep(1)  # Sleep to avoid constant checking

    async def extend_palindrome_next_level(self, palindrome):
        """Try to extend the palindrome using anagram words"""
        anagram_sources = [
            feed.subs_anagrams,
            feed.verb_anagrams,
            feed.adj_anagrams,
            feed.long_anagrams
        ]

        # For each source of anagrams, try to extend the palindrome
        for source in anagram_sources:
            for word in source:
                # Create new palindromes by adding the word at both ends
                extended_start = word + " " + palindrome + " " + word[::-1]  # Add at both ends
                feed.new_palindromes.append(extended_start)

    def format_list(self, data_list, width=80):
        """ Format list to display with a specified width per row """
        # Join the list into a single string and wrap it to the specified width
        return '\n'.join(textwrap.wrap(', '.join(data_list), width))

    async def print_status(self):
        global fail_counter
        """ Print message and status on the same line """
        while True:
            formatted_palindromes = self.format_list(feed.new_palindromes)
            found_counter = len(feed.new_palindromes)
            self.status = (f"Tries: {fail_counter}  -  Found: {found_counter}  -  Currently in: "
                           f"{begin_word}\nFound palindromes:\n{formatted_palindromes}\n")
            if self.cancel_requested:
                if self.debug:
                    print("Generation interrupted! (print_status)")
                break
            if self.debug:
                print(f"Tries: {fail_counter} Currently in: {begin_word}\nFound palindromes:\n{formatted_palindromes}\n", flush=True)
            await asyncio.sleep(1)


class GameInstructions(QDialog):
    """
    This class is for game instructions screen
    """
    TXT_WINDOWS_TITLE = "Peliohjeita"

    def __init__(self, palindromes_file="palindromes.json"):
        super().__init__()
        self.instructions_ui = Ui_game_instructions_Dialog()  #
        self.instructions_ui.setupUi(self)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)


class InspectDialog(QDialog):
    """
    This class is for inspecting learned palindromes.

    args: palindromes-file
    """
    TXT_WINDOWS_TITLE = "Palindromien tarkastelu"

    def __init__(self, palindromes_file="palindromes.json"):
        super().__init__()
        self.inspect_ui = Ui_inspect_Dialog()  # Tämä oletetaan olevan erillinen UI-luokka
        self.inspect_ui.setupUi(self)
        self.setup_ui()

        self.palindromes_listview_model = QtGui.QStandardItemModel()

        if palindromes_file and os.path.exists(palindromes_file):
            try:
                with open(palindromes_file, 'r', encoding='utf-8') as f:
                    self.palindromes = json.load(f)
            except FileNotFoundError as e:
                logging.error(f"File not found: {palindromes_file}")
                raise e

        self.palindrome_list = [value for key, value in self.palindromes.items()]
        self.inspect_ui.palindromes_listView.setModel(self.palindromes_listview_model)

    def selected_text(self):
        input_text = self.inspect_ui.input_word_lineEdit.text().strip()
        filtered_palindromes = [p for p in self.palindrome_list if input_text.lower() in p.lower()]
        self.inspect_ui.found_lcdNumber.display(len(filtered_palindromes))
        self.palindromes_listview_model.clear()
        for palindrome in filtered_palindromes:
            self.palindromes_listview_model.appendRow(QtGui.QStandardItem(palindrome))

    def setup_ui(self):
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)
        self.inspect_ui.input_word_lineEdit.textChanged.connect(self.selected_text)
        self.inspect_ui.input_word_lineEdit.setCursorPosition(0)
        self.inspect_ui.input_word_lineEdit.setFocus()
        self.inspect_ui.found_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super(MatplotlibCanvas, self).__init__(self.fig)


class GraphDialog(QDialog):
    """
        This class is for vector visualization.

        args: word_vectors model, words selected in the user form

    """

    TXT_MODEL_VECTORS = "Palindromimallin vektorit"
    TXT_WINDOWS_TITLE = "Mallin visualisointi - Zoomaa lähemmäs!"

    def __init__(self, word_vectors, words, parent=None):
        super(GraphDialog, self).__init__(parent)

        self.graph_ui = Ui_graph_Dialog()
        self.graph_ui.setupUi(self)
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)

        self.canvas = MatplotlibCanvas(self)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.graph_ui.graph_layout.addWidget(self.toolbar)  # Add toolbar first
        self.graph_ui.graph_layout.addWidget(self.canvas)  # Then canvas!

        self.show_pca_graph(word_vectors, words)

    def show_pca_graph(self, word_vectors, words):
        # PCA visualization of all word vectors
        pca = PCA(n_components=2)
        word_vecs_2d = pca.fit_transform(word_vectors)

        self.canvas.ax.clear()
        self.canvas.ax.set_title(self.TXT_MODEL_VECTORS)
        self.canvas.ax.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1])

        # Annotate words on the plot
        for i, word in enumerate(words):
            self.canvas.ax.annotate(word, xy=(word_vecs_2d[i, 0], word_vecs_2d[i, 1]))

        self.canvas.draw()


class GENERATEDialog(QDialog):
    """
    This dialog is for new palindrome generation and converting found palindromes into palindromes.json file

    Initializes palindrome generator object. Note! Asynchronous code within!

    args: debug

    """

    TXT_WINDOWS_TITLE= "Palindromien generointi"
    TXT_FILE_CHOSEN = "Valitsit tiedoston: "
    TXT_READY_TO_START = " .. voit aloittaa generoinnin"
    TXT_SELECT_FILE = "- valitse tiedosto -"
    TXT_CANCELLED = "Keskeytit generoinnin!"

    def __init__(self, parent=None, debug=False):
        super(GENERATEDialog, self).__init__(parent)
        self.debug=debug
        self.generator_ui = Ui_generate_palindromes_Dialog()
        self.generator_ui.setupUi(self)
        self.setup_ui()
        self.maker = None
        self.maker_status = "Odottaa aloittamista..."
        self.selected_file = None
        self.new_file = None
        self.selected_wordlist = None
        self.cancel_requested = False


    def convert_csv(self):
        # Note! All found new_ files must exist prior call!
        self.maker = PalindromeMaker(debug=False)
        self.maker.convert_new_csv_to_json()
        self.generator_ui.status_Right_label.setText(self.maker.status)

    async def generate_palindromes(self):
        self.maker = PalindromeMaker(debug=False)
        self.maker.new_file = self.new_file
        self.maker.chosen_wordlist = self.selected_wordlist

        # Start both background tasks
        self.generator_ui.cancel_generation_pushButton.setEnabled(True)
        self.generator_ui.cancel_generation_pushButton.setStyleSheet("background-color: green; color: white;")
        status_print_task = asyncio.create_task(self.update_status_async())  # Updates form status
        print_task = asyncio.create_task(self.maker.print_status())  # Create the periodic print task

        try:
            await self.maker.make_palindromes_for_learning(200000)  # This will run the main learning process
        except asyncio.CancelledError:
            self.generator_ui.status_Right_label.setText(self.TXT_CANCELLED)
            self.maker.save_progress() # save even if cancelled!
            print_task.cancel()
            status_print_task.cancel()
        finally:
            # Cancel the background tasks once done
            self.maker.save_progress()  # save even if cancelled!
            self.generator_ui.cancel_generation_pushButton.setEnabled(False)
            self.generator_ui.cancel_generation_pushButton.setStyleSheet("background-color: grey; color: white;")
            self.generator_ui.generate_Button.setEnabled(True)


    def cancel_generation(self):
        """If user press Cancel Generation (Peru Generointi), this function is acticated."""
        self.cancel_requested = True
        # Cancel asynchronous task, inform status
        self.generator_ui.status_Right_label.setText("Interrupting task...")
        # Send interrupt signal to the generator
        self.maker.cancel_generation()

    async def update_status_async(self):
        while True:
            # Update form with status
            if self.maker:
                self.maker_status = self.maker.status
            self.generator_ui.status_Right_label.setText(self.maker_status)
            await asyncio.sleep(1)  # update once a second

    def on_file_selected(self):
        self.selected_file = self.generator_ui.filenames_comboBox.currentText()
        self.generator_ui.status_labelLeft.setText(
            f'<span style="color: black;">{self.TXT_FILE_CHOSEN}</span>'
            f'<span style="color: blue; font-weight: bold;">{self.selected_file}</span>'
            f'<span style="color: black;">{self.TXT_READY_TO_START}</span>'
        )

        file_map = {
            verbs_file: new_verb_palindromes_file,
            adjectives_file: new_adj_palindromes_file,
            substantives_file: new_subs_palindromes_file,
            long_sentences_file: new_long_text_palindromes_file
        }

        wordlist_map = {
            verbs_file: feed.clean_verbs,
            adjectives_file: feed.clean_adjectives,
            substantives_file: feed.clean_substantives,
            long_sentences_file: feed.extracted_words
        }

        if self.selected_file in file_map:
            self.new_file = file_map[self.selected_file]
            self.selected_wordlist = wordlist_map[self.selected_file]
            self.generator_ui.generate_Button.setEnabled(True)
            self.generator_ui.generate_Button.setStyleSheet("background-color: green; color: white;")
        else:
            self.generator_ui.generate_Button.setEnabled(False)
            self.generator_ui.generate_Button.setStyleSheet("background-color: grey; color: white;")

    def setup_ui(self):
        # Initial settings for the form
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)
        self.generator_ui.generate_Button.setEnabled(False)
        self.generator_ui.generate_Button.setStyleSheet("background-color: grey; color: white;")
        self.generator_ui.convertButton.setDisabled(True)
        self.generator_ui.cancel_generation_pushButton.setEnabled(False)
        self.generator_ui.cancel_generation_pushButton.setStyleSheet("background-color: grey; color: white;")
        self.generator_ui.filenames_comboBox.addItem(self.TXT_SELECT_FILE)
        if os.path.exists(verbs_file):
            self.generator_ui.filenames_comboBox.addItem(verbs_file)
        if os.path.exists(adjectives_file):
            self.generator_ui.filenames_comboBox.addItem(adjectives_file)
        if os.path.exists(substantives_file):
            self.generator_ui.filenames_comboBox.addItem(substantives_file)
        if os.path.exists(long_sentences_file):
            self.generator_ui.filenames_comboBox.addItem(long_sentences_file)
        self.generator_ui.filenames_comboBox.currentIndexChanged.connect(self.on_file_selected)
        self.generator_ui.generate_Button.clicked.connect(lambda: asyncio.create_task(self.generate_palindromes()))
        self.generator_ui.cancel_generation_pushButton.clicked.connect(self.cancel_generation)
        self.generator_ui.convertButton.clicked.connect(self.convert_csv)

        if (os.path.exists(new_verb_palindromes_file) and os.path.exists(new_adj_palindromes_file)
                and os.path.exists(new_subs_palindromes_file) and os.path.exists(new_long_text_palindromes_file)):
            self.generator_ui.convertButton.setDisabled(False)
            self.generator_ui.convertButton.setStyleSheet("background-color: green; color: white;")
        else:
            self.generator_ui.convertButton.setStyleSheet("background-color: grey; color: white;")


class NLTKDialog(QDialog):
    TXT_MODEL_INSPECTION = "Mallin tarkastelu"
    TXT_MODEL_WORDS = "Mallissa sanoja:"
    TXT_PLT_TOPIC = "Palindromimallin vektorit"
    TXT_NOT_FOUND="Ei löydy"
    TXT_WINDOWS_TITLE= "Mallin tarkastelu"

    def __init__(self, model, parent=None):
        super(NLTKDialog, self).__init__(parent)
        self.model = model
        self.nltk_ui = Ui_NLTKDialog()
        self.nltk_ui.setupUi(self)
        self.setup_ui()

        self.vectors_listview_model = QtGui.QStandardItemModel()
        # Fetch the words learned by the model and their vectors
        self.words = list(self.model.wv.index_to_key)
        self.word_vectors = self.model.wv[self.words]

        # Display the number of words in the model
        self.nltk_ui.words_total_lcdNumber.display(len(self.model.wv.index_to_key))
        self.nltk_ui.showVectorsButton.clicked.connect(self.show_vectors)
        self.nltk_ui.visualize_pushButton.clicked.connect(self.show_visualization)


    def show_vectors(self):
        # Show the specific input words and their vectors
        self.vectors_listview_model.clear()
        input_words = self.nltk_ui.input_Word.text().lower().split()
        if input_words is not None:
            for text in input_words:
                if text in self.model.wv:
                    vector = self.model.wv[text]
                    self.vectors_listview_model.appendRow((QtGui.QStandardItem(f"{text}: {vector}")))
                else:
                    self.vectors_listview_model.appendRow((QtGui.QStandardItem(self.TXT_NOT_FOUND)))
            self.nltk_ui.vector_listView.setModel(self.vectors_listview_model)

    def show_visualization(self):
        graph_dialog = GraphDialog(self.word_vectors, self.words, self)
        graph_dialog.exec()

    def setup_ui(self):
        # Initial settings for the form
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)
        self.nltk_ui.words_total_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.nltk_ui.words_total_lcdNumber.setFixedWidth(100)


class MainWindow(QMainWindow):
    """
    This is main class and window for QT6 forms. Run this first!
    If you would like to make new model_file, just delete old and it will be generated again.

    args: palindromes.json and palindrome_word2vec.model are hardcoded to this code, not from runtimeconfig.json!

    """

    TXT_WINDOWS_TITLE = "Palindromipeli 0.32"
    TXT_INFO = "Palindromeja ladattu: "
    TXT_WRITE_LEFT = "Kirjoita vasemmalle..."
    TXT_WRITE_CENTER = "Kirjoita keskelle..."
    TXT_WRITE_RIGHT = "Kirjoita oikealle..."
    TXT_SUGGESTIONS = "Ehdotelmat (samankaltaisuus %):"
    TXT_NO_SUGGESTIONS = "Ei suosituksia saatavilla."
    TXT_FEEDBACK = "Onko palindromi?"
    TXT_IS_PALINDROME = "on palindromi! Upeaa!"
    TXT_ISNOT_PALINDROME = " ei ole palindromi :("
    TXT_SIMILARITY = "samankaltaisuus %:"
    TXT_WORD_FOUND = "- on sanastossa!"
    TXT_WORD_NOT_FOUND = "- ei löydy"
    TXT_WORD_IN_PALINDROMES = "Sana palindromeissa:\n"
    TXT_GAME_INSTRUCTIONS =  "Peliohjeet"
    TXT_ABOUT_NLTK_MODEL = "NLTK-mallin tarkastelu"
    TXT_GENERATE_PALINDROMES = "Palindromien generointi"
    TXT_INSPECT = "Tarkastele palindromeja"

    def __init__(self, palindromes_file="palindromes.json", model_file="palindrome_word2vec.model"):
        super().__init__()
        self.main_ui = Ui_first_window()
        self.main_ui.setupUi(self)

        # Left, Center and Right ListView models for similar words
        self.left_listview_model = QtGui.QStandardItemModel()
        self.center_listview_model = QtGui.QStandardItemModel()
        self.right_listview_model = QtGui.QStandardItemModel()

        # Left, Center and Right ListView models for palindromes
        self.left_palindromes_listview_model = QtGui.QStandardItemModel()
        self.center_palindromes_listview_model = QtGui.QStandardItemModel()
        self.right_palindromes_listview_model = QtGui.QStandardItemModel()

        self.combined_text = ""
        self.begin_word = ""
        self.is_mirrored_text = False
        self.left_match_word = ""
        self.right_match_word = ""
        self.center_match_word = ""
        self.was_palindrome = False
        self.model_window_active = False
        self.iterations=0
        self.found_palindromes_count = 0
        self.found_palindromes = []
        self.found_palindromes_multiplier = 10
        self.found_new_palindromes_count = 0
        self.found_new_palindromes = []
        self.found_new_palindromes_multiplier = 1000
        self.total_points = 0
        self.setup_ui()

        # Make set lists (faster), make words lower case
        self.words = {word.lower() for word in
                      set(feed.clean_verbs) | set(feed.clean_adjectives) |
                      set(feed.clean_substantives) | set(feed.extracted_words)}

        if palindromes_file and os.path.exists(palindromes_file):
            try:
                with open(palindromes_file, 'r', encoding='utf-8') as f:
                    self.palindromes = json.load(f)
            except FileNotFoundError as e:
                logging.error(f"File not found: {palindromes_file}")
                raise e

        # Make list of palindromes
        self.palindrome_list = [value for key, value in self.palindromes.items()]

        if not os.path.exists(model_file):
            # Tokenize words
            nltk.download('punkt')
            self.palindrome_tokens = [word_tokenize(self.palindromes.lower()) for self.palindrome in
                                      self.palindrome_list]

            """            
            Alternative learning model - keep this for testing
            
            # Word2Vec-mallin rakentaminen
            self.model = Word2Vec(sentences=self.palindrome_tokens, vector_size=100, window=5, min_count=1, workers=4)
            self.model.train(self.palindrome_tokens, total_examples=len(self.palindrome_tokens), epochs=20)
            self.model.save(model_file)
            """

            # building a FastText model
            self.wordlist_model = FastText(sentences=self.palindrome_tokens, vector_size=100, window=5, min_count=1, workers=4)
            self.wordlist_model.train(self.palindrome_tokens, total_examples=len(self.palindrome_tokens), epochs=30)
            self.wordlist_model.save(model_file)

        else:
            try:
                self.wordlist_model = Word2Vec.load(model_file)
            except FileNotFoundError as e:
                logging.error(f"File not found: {model_file}")
                raise e

        # Palindrome count
        self.main_ui.palindromes_lcdNumber.display(len(self.palindrome_list))

        self.suggestion_label = QLabel(self.TXT_SUGGESTIONS)


    def setup_ui(self):
        self.setWindowTitle(self.TXT_WINDOWS_TITLE)
        self.main_ui.palindromes_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.main_ui.palindromes_lcdNumber.setFixedWidth(80)

        self.main_ui.total_score_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.main_ui.total_score_lcdNumber.setFixedWidth(100)

        self.main_ui.iterations_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.main_ui.iterations_lcdNumber.setFixedWidth(80)

        self.main_ui.new_palindromes_points_lcdNumber.setStyleSheet("""QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.main_ui.new_palindromes_points_lcdNumber.setFixedWidth(100)

        self.main_ui.found_palindromes_points_lcdNumber.setStyleSheet(
            """QLCDNumber {background-color: rgb(0, 85, 0);}""")
        self.main_ui.found_palindromes_points_lcdNumber.setFixedWidth(100)

        # Left input field
        self.main_ui.left_input.setPlaceholderText(self.TXT_WRITE_LEFT)
        self.main_ui.left_input.textChanged.connect(self.mirror_left_to_right)

        # Center input field
        self.main_ui.center_input.setPlaceholderText(self.TXT_WRITE_CENTER)
        self.main_ui.center_input.textChanged.connect(self.update_middle)

        # Right input field
        self.main_ui.right_input.setPlaceholderText(self.TXT_WRITE_RIGHT)
        self.main_ui.right_input.textChanged.connect(self.mirror_right_to_left)

        # Mirror button
        self.main_ui.mirror_left_and_rightcheckBox.setEnabled(True)
        self.main_ui.mirror_left_and_rightcheckBox.clicked.connect(self.mirror_on_off)

        # Learning button
        tools_menu = QMenu(self.main_ui.settings_Button)

        # Actions for the learning button
        action1 = tools_menu.addAction(self.TXT_GAME_INSTRUCTIONS)
        action2 = tools_menu.addAction(self.TXT_ABOUT_NLTK_MODEL)
        action3 = tools_menu.addAction(self.TXT_GENERATE_PALINDROMES)
        action4 = tools_menu.addAction(self.TXT_INSPECT)

        # Menu items connection to methods
        # noinspection PyUnresolvedReferences
        action1.triggered.connect(self.instructions_menu)
        # noinspection PyUnresolvedReferences
        action2.triggered.connect(self.nltk_model_menu)
        # noinspection PyUnresolvedReferences
        action3.triggered.connect(self.generate_palindromes_menu)
        # noinspection PyUnresolvedReferences
        action4.triggered.connect(self.inspect_menu)

        # Add menu
        self.main_ui.settings_Button.setMenu(tools_menu)

    def instructions_menu(self):
        model_window = GameInstructions()
        model_window.exec()

    def nltk_model_menu(self):
        # Pass model to subclass and create or close and show the ModelWindow
        model_window = NLTKDialog(self.wordlist_model, self)
        model_window.exec()

    def generate_palindromes_menu(self):
        model_window = GENERATEDialog()
        model_window.exec()

    def inspect_menu(self):
        model_window = InspectDialog()
        model_window.exec()

    def mirror_on_off(self):
        if self.main_ui.mirror_left_and_rightcheckBox.isChecked():
            self.is_mirrored_text = True
        else:
            self.is_mirrored_text = False

    def mirror_left_to_right(self):
        """ Peilataan vasemman kentän teksti oikeaan kenttään yhden kerran. """
        text = self.main_ui.left_input.text().lower()
        if len(text)>1:
            self.main_ui.right_input.blockSignals(True)  # Estetään takaisinpeilaus
            if self.is_mirrored_text:
                self.main_ui.right_input.setText(text[::-1])
            self.main_ui.right_input.blockSignals(False)
            self.run_async_task()  # Suoritetaan asynkroninen tehtävä

    def mirror_right_to_left(self):
        """ Peilataan oikean kentän teksti vasempaan kenttään. """
        text = self.main_ui.right_input.text().lower()
        if len(text) > 1:
            self.main_ui.left_input.blockSignals(True)  # Estetään takaisinpeilaus
            if self.is_mirrored_text:
                self.main_ui.left_input.setText(text[::-1])
            self.main_ui.left_input.blockSignals(False)
            self.run_async_task()  # Suoritetaan asynkroninen tehtävä

    def update_middle(self):
        self.main_ui.center_input.lower()
        if len(self.main_ui.left_input.text()) > 1:
            self.mirror_left_to_right()
        if len(self.main_ui.right_input.text()) >1:
            self.mirror_right_to_left()
        self.run_async_task()  # Suoritetaan asynkroninen tehtävä


    def is_palindrome_combined(self, left_text, center_text, right_text):
        """Yhdistetään vasen, keski ja oikea kenttä ja tarkistetaan, onko ne palindromi ja lasketaan pisteitä."""
        self.main_ui.result_palindrome_label.setText("...")
        if len(left_text) > 1:
            self.begin_word =  left_text.split()[0]
        else:
            return False

        self.combined_text = left_text.lower() + center_text.lower() + right_text.lower()

        # Poistetaan välilyönnit ja erikoismerkit tarkistusta varten
        cleaned_text = ''.join([c for c in self.combined_text.lower() if c.isalnum()])

        # Tarkista, onko palindromi
        if cleaned_text != cleaned_text[::-1]:
            self.main_ui.result_palindrome_label.setText(self.combined_text + " " + self.TXT_ISNOT_PALINDROME)
            self.main_ui.result_palindrome_label.repaint()
            self.was_palindrome = False
            return False

        # Jaetaan yhdistetty teksti sanoiksi
        words_in_combined_text = []
        words_in_combined_text.extend(left_text.split())
        words_in_combined_text.extend(center_text.split())
        words_in_combined_text.extend(right_text.split())

        # Tarkista, löytyykö jokainen sana sanalistasta
        all_found = True  # Oletetaan, että kaikki sanat löytyvät

        for word in words_in_combined_text:
            word = word.lower()  # Varmistetaan, että verrataan pienillä kirjaimilla
            if word in self.words:
                pass
            else:
                all_found = False  # Jos edes yksi sana ei löydy, asetetaan all_found vääräksi

        # Päivitetään näyttö, jos kaikki sanat löytyivät, lisätään pisteitä
        if all_found:

            search_text_with_spaces = f"{left_text} {center_text} {right_text}"

            # Add palindrome to the list and give points if not exists already and suggestions are off
            if (search_text_with_spaces in self.palindrome_list
                    and not self.main_ui.show_matching_palindromes_checkBox.isChecked()
                    and self.combined_text not in self.found_palindromes):
                self.found_palindromes_count +=1
                self.found_palindromes.append(self.combined_text)

            # Ok if suggestions are displayed when hunting totally new palindrome
            elif ((search_text_with_spaces not in self.palindrome_list)
                  and (self.combined_text not in self.found_new_palindromes)):
                self.found_new_palindromes_count +=1
                self.found_new_palindromes.append(self.combined_text)

            self.total_points = ((self.found_new_palindromes_count * self.found_new_palindromes_multiplier)
                                 + (self.found_palindromes_count * self.found_palindromes_multiplier))

            self.main_ui.new_palindromes_points_lcdNumber.display(self.found_new_palindromes_count
                                                                  * self.found_new_palindromes_multiplier)
            self.main_ui.found_palindromes_points_lcdNumber.display(self.found_palindromes_count
                                                                    * self.found_palindromes_multiplier)
            self.main_ui.total_score_lcdNumber.display(self.total_points)
        else:
            self.main_ui.result_palindrome_label.setWordWrap(True)
            self.main_ui.result_palindrome_label.setText(left_text + " " + center_text + " " +
                                                         right_text + " " + " " + self.TXT_ISNOT_PALINDROME)
            self.main_ui.result_palindrome_label.repaint()
            self.was_palindrome = False

        self.main_ui.result_palindrome_label.setWordWrap(True)
        self.main_ui.result_palindrome_label.setText(left_text + " " + center_text + " "
                                                     + right_text + " " + self.TXT_IS_PALINDROME)

        self.main_ui.result_palindrome_label.repaint()
        self.was_palindrome = True

        return all_found


    @asyncSlot()
    async def run_async_task(self):
        self.left_listview_model.clear()
        self.center_listview_model.clear()
        self.right_listview_model.clear()
        self.is_palindrome_combined(self.main_ui.left_input.text(), self.main_ui.center_input.text(), self.main_ui.right_input.text())
        if self.was_palindrome:
            self.main_ui.result_palindrome_label.setStyleSheet('color: red; font-size: 18px; background-color: rgb(249, 240, 107);')
        else:
            self.main_ui.result_palindrome_label.setStyleSheet('color: blue; font-size: 14px;background-color: rgb(222, 221, 218);')
        # Show suggestions based on input (left, center and right)
        left_suggestions =await self.recommend_words_for_palindrome(self.main_ui.left_input.text().lower(),
                                                                    self.wordlist_model)
        await self.show_suggestions(left_suggestions, "left")
        center_suggestions = await self.recommend_words_for_palindrome(self.main_ui.center_input.text().lower(),
                                                                       self.wordlist_model)
        await self.show_suggestions(center_suggestions, "center")
        right_suggestions = await self.recommend_words_for_palindrome(self.main_ui.right_input.text().lower(),
                                                                      self.wordlist_model)
        await self.show_suggestions(right_suggestions, "right")

        # Check if word is found in words lists - adjusts self-variables
        await self.show_if_word_found()
        # Show matching palindromes
        if self.main_ui.show_matching_palindromes_checkBox.isChecked():
            await self.show_existing_palindromes()
        else:
            # Listataan vain kerran
            self.left_palindromes_listview_model.clear()
            self.center_palindromes_listview_model.clear()
            self.right_palindromes_listview_model.clear()
            await self.update_listview()
        self.iterations += 1
        self.main_ui.iterations_lcdNumber.display(self.iterations)

        await asyncio.sleep(0)

    async def show_if_word_found(self):
        """ Display if left, center or right word is found from vocabulary """
        if len(self.main_ui.left_input.text())>0:
            first_word_left = self.main_ui.left_input.text().split()[0]
            if first_word_left.lower() in self.words:
                self.main_ui.left_found_label.setText(first_word_left + " " + self.TXT_WORD_FOUND)
                self.main_ui.left_found_label.setStyleSheet('color: green; font-size: 16px; font-family: Arial;')
            else:
                self.main_ui.left_found_label.setText(first_word_left + " " + self.TXT_WORD_NOT_FOUND)
                self.main_ui.left_found_label.setStyleSheet('color: red; font-size: 12px; font-family: Arial;')
        if len(self.main_ui.center_input.text())>0:
            first_word_center = self.main_ui.center_input.text().split()[0]
            if first_word_center.lower() in self.words:
                self.main_ui.center_found_label.setText(first_word_center + " " + self.TXT_WORD_FOUND)
                self.main_ui.center_found_label.setStyleSheet('color: green; font-size: 16px; font-family: Arial;')
            else:
                self.main_ui.center_found_label.setText(first_word_center + " " + self.TXT_WORD_NOT_FOUND)
                self.main_ui.center_found_label.setStyleSheet('color: red; font-size: 12px; font-family: Arial;')
        if len(self.main_ui.right_input.text())>0:
            first_word_right = self.main_ui.right_input.text().split()[0]
            if first_word_right.lower() in self.words:
                self.main_ui.right_found_label.setText(first_word_right + " " + self.TXT_WORD_FOUND)
                self.main_ui.right_found_label.setStyleSheet('color: green; font-size: 16px; font-family: Arial;')
            else:
                self.main_ui.right_found_label.setText(first_word_right + " " + self.TXT_WORD_NOT_FOUND)
                self.main_ui.right_found_label.setStyleSheet('color: red; font-size: 12px; font-family: Arial;')

        await asyncio.sleep(0)

    async def recommend_words_for_palindrome(self, text, model, topn=5):
        """ Suosittelee sanoja palindromin muokkaamiseen opetetun mallin perusteella. """

        if text =="":
            return [self.TXT_NO_SUGGESTIONS]
        elif text in model.wv:
            suggestions =model.wv.most_similar(text, topn=topn)
            return suggestions
        else:
            return [self.TXT_NO_SUGGESTIONS]


    async def show_suggestions(self, suggestions, location):
        """Näytetään suositellut sanat pohjautuen malliin. """

        if isinstance(suggestions, list) and isinstance(suggestions[0], tuple):
            suggestion_text = "\n".join([f"{word} ({similarity * 100:.1f})"
                                         for word, similarity in suggestions])
        else:
            suggestion_text = self.TXT_NO_SUGGESTIONS

        if location == "left":
            self.left_listview_model.clear()
            self.left_listview_model.appendRow(QtGui.QStandardItem(self.TXT_SUGGESTIONS + f"\n\n{suggestion_text}"))
        elif location == "center":
            self.center_listview_model.clear()
            self.center_listview_model.appendRow(
                QtGui.QStandardItem(self.TXT_SUGGESTIONS + f"\n\n{suggestion_text}"))
        elif location == "right":
            self.right_listview_model.clear()
            self.right_listview_model.appendRow(
                    QtGui.QStandardItem(self.TXT_SUGGESTIONS + f"\n\n{suggestion_text}"))

        await self.update_listview()
        await asyncio.sleep(0)


    async def show_existing_palindromes(self):
        """Näytetään suositellut palindromit joissa sana esiintyy. Huom! Tässä listassa ensimmäisenä ehdotelmat!"""

        # Listataan vain kerran
        self.left_palindromes_listview_model.clear()
        self.center_palindromes_listview_model.clear()
        self.right_palindromes_listview_model.clear()

        if self.main_ui.show_matching_palindromes_checkBox.isChecked():

            self.left_palindromes_listview_model.appendRow(QtGui.QStandardItem(self.TXT_WORD_IN_PALINDROMES))
            self.center_palindromes_listview_model.appendRow(QtGui.QStandardItem(self.TXT_WORD_IN_PALINDROMES))
            self.right_palindromes_listview_model.appendRow(QtGui.QStandardItem(self.TXT_WORD_IN_PALINDROMES))

            left_matching_palindromes = []
            center_matching_palindromes = []
            right_matching_palindromes = []

            for sentence in self.palindrome_list:
                if self.main_ui.left_input.text() in sentence.split():
                    left_matching_palindromes.append(sentence)
                if self.main_ui.center_input.text() in sentence.split():
                    center_matching_palindromes.append(sentence)
                if self.main_ui.right_input.text() in sentence.split():
                    right_matching_palindromes.append(sentence)

            for palindrome in left_matching_palindromes[:5]:
                self.left_palindromes_listview_model.appendRow(QtGui.QStandardItem(palindrome))

            for palindrome in center_matching_palindromes[:5]:
                self.center_palindromes_listview_model.appendRow(QtGui.QStandardItem(palindrome))

            for palindrome in right_matching_palindromes[:5]:
                self.right_palindromes_listview_model.appendRow(QtGui.QStandardItem(palindrome))

            await self.update_listview()

        await asyncio.sleep(0)


    async def update_listview(self):
        """ Updates listview left, center and right with word suggestions and palindromes"""

        self.main_ui.left_listView.setModel(self.left_listview_model)
        self.main_ui.center_listView.setModel(self.center_listview_model)
        self.main_ui.right_listView.setModel(self.right_listview_model)

        self.main_ui.palindromes_left.setModel(self.left_palindromes_listview_model)
        self.main_ui.palindromes_center.setModel(self.center_palindromes_listview_model)
        self.main_ui.palindromes_right.setModel(self.right_palindromes_listview_model)

        # self.ui.left_listView.update()  # tai self.ui.left_listView.repaint()
        # self.ui.center_listView.update()  # tai self.ui.center_listView.repaint()
        # self.ui.right_listView.update()  # tai self.ui.right_listView.repaint()




# Load vocabularies for FEEDER class
feed = FEEDER(debug=False)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Fusion on yhtenäinen tyyli, joka toimii hyvin eri alustoilla

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()
