"""
DOCSTRING
"""
import collections
import copy
import itertools
import keras.layers.core
import keras.layers.recurrent
import keras.models
import music21
import numpy
import random.choice
import sys

class GenerateMusic:
    """
    DOCSTRING
    """
    def __generate_grammar(
        self,
        model,
        corpus,
        abstract_grammars,
        values,
        val_indices,
        indices_val,
        max_len,
        max_tries,
        diversity):
        """
        Helper function which uses the given model to generate a grammar sequence 
        from a given corpus, indices_val (mapping), abstract_grammars (list), 
        and diversity floating-point value.
        """
        curr_grammar = ''
        start_index = numpy.random.randint(0, len(corpus) - max_len)
        sentence = corpus[start_index: start_index + max_len]
        running_length = 0.0
        while running_length <= 4.1:
            x = numpy.zeros((1, max_len, len(values)))
            for t, val in enumerate(sentence):
                if (not val in val_indices): print(val)
                x[0, t, val_indices[val]] = 1.0
            next_val = self.__predict(model, x, indices_val, diversity)
            if (running_length < 0.00001):
                tries = 0
                while (next_val.split(',')[0] == 'R' or len(next_val.split(',')) != 2):
                    if tries >= max_tries:
                        print('Gave up on first note generation after {} tries'.format(max_tries))
                        rand = numpy.random.randint(0, len(abstract_grammars))
                        next_val = abstract_grammars[rand].split(' ')[0]
                    else:
                        next_val = self.__predict(model, x, indices_val, diversity)
                    tries += 1
            sentence = sentence[1:] 
            sentence.append(next_val)
            if (running_length > 0.00001):
                curr_grammar += ' '
            curr_grammar += next_val
            length = float(next_val.split(',')[1])
            running_length += length
        return curr_grammar

    def __predict(self, model, x, indices_val, diversity):
        """
        Helper function to generate a predicted value from a given matrix.
        """
        preds = model.predict(x, verbose=0)[0]
        next_index = self.__sample(preds, diversity)
        next_val = indices_val[next_index]
        return next_val

    def __sample(self, a, temperature=1.0):
        """
        Helper function to sample an index from a probability array.
        """
        a = numpy.log(a) / temperature
        a = numpy.exp(a) / numpy.sum(numpy.exp(a))
        return numpy.argmax(numpy.random.multinomial(1, a, 1))

    def generate(self, data_fn, out_fn, N_epochs):
        """
        Generates musical sequence based on the given data filename and settings.
        Plays then stores (MIDI file) the generated output.
        """
        max_len = 20
        max_tries = 1000
        diversity = 0.5
        bpm = 130
        chords, abstract_grammars = Preprocess.get_musical_data(data_fn)
        corpus, values, val_indices, indices_val = Preprocess.get_corpus_data(abstract_grammars)
        print('corpus length:', len(corpus))
        print('total # of values:', len(values))
        model = LTSM.build_model(
            corpus=corpus, val_indices=val_indices,  max_len=max_len, N_epochs=N_epochs)
        out_stream = stream.Stream()
        curr_offset = 0.0
        loopEnd = len(chords)
        for loopIndex in range(1, loopEnd):
            curr_chords = stream.Voice()
            for j in chords[loopIndex]:
                curr_chords.insert((j.offset % 4), j)
            curr_grammar = self.__generate_grammar(
                model=model, corpus=corpus, abstract_grammars=abstract_grammars, 
                values=values, val_indices=val_indices, indices_val=indices_val, 
                max_len=max_len, max_tries=max_tries, diversity=diversity)
            curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')
            curr_grammar = QA.prune_grammar(curr_grammar)
            curr_notes = Grammar.unparse_grammar(curr_grammar, curr_chords)
            curr_notes = QA.prune_notes(curr_notes)
            curr_notes = QA.clean_up_notes(curr_notes)
            print('After pruning: %s notes' % (
                len([i for i in curr_notes if isinstance(i, note.Note)])))
            for m in curr_notes:
                out_stream.insert(curr_offset + m.offset, m)
            for mc in curr_chords:
                out_stream.insert(curr_offset + mc.offset, mc)
            curr_offset += 4.0
        out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))
        play = lambda x: midi.realtime.StreamPlayer(x).play()
        play(out_stream)
        mf = midi.translate.streamToMidiFile(out_stream)
        mf.open(out_fn, 'wb')
        mf.write()
        mf.close()

    def main(self, args):
        """
        Runs generate function (generating, playing, then storing a musical sequence),
        with the default Metheny file.
        """
        try:
            N_epochs = int(args[1])
        except:
            N_epochs = 128
        data_fn = 'midi/original_metheny.midi' 
        out_fn = 'midi/deepjazz_on_metheny...{}'.format(str(N_epochs))
        if (N_epochs == 1):
            out_fn += '_epoch.midi'
        else:
            out_fn += '_epochs.midi'
        self.generate(data_fn, out_fn, N_epochs)

class Grammar:
    """
    DOCSTRING
    """
    def __is_scale_tone(self, chord, note):
        """
        Helper function to determine if a note is a scale tone.
        Generate all scales that have the chord notes then check if note is in names.
        Derive major or minor scales (minor if 'other') based on the quality of the chord.
        """
        scaleType = scale.DorianScale()
        if chord.quality == 'major':
            scaleType = scale.MajorScale()
        scales = scaleType.derive(chord)
        allPitches = list(set([pitch for pitch in scales.getPitches()]))
        allNoteNames = [i.name for i in allPitches]
        noteName = note.name
        return (noteName in allNoteNames)

    def __is_approach_tone(self, chord, note):
        """
        Helper function to determine if a note is an approach tone.
        Determine if note is +/- 1 a chord tone.
        """
        for chordPitch in chord.pitches:
            stepUp = chordPitch.transpose(1)
            stepDown = chordPitch.transpose(-1)
            if (note.name == stepDown.name or 
                note.name == stepDown.getEnharmonic().name or
                note.name == stepUp.name or
                note.name == stepUp.getEnharmonic().name):
                    return True
        return False

    def __is_chord_tone(self, lastChord, note):
        """
        Helper function to determine if a note is a chord tone.
        """
        return (note.name in (p.name for p in lastChord.pitches))

    def __generate_chord_tone(self, lastChord):
        """
        Helper function to generate a chord tone.
        """
        lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
        return note.Note(random.choice(lastChordNoteNames))

    def __generate_scale_tone(self, lastChord):
        """
        Helper function to generate a scale tone.
        Derive major or minor scales (minor if 'other') based on the quality of the lastChord.
        """
        scaleType = scale.WeightedHexatonicBlues()
        if lastChord.quality == 'major':
            scaleType = scale.MajorScale()
        scales = scaleType.derive(lastChord)
        allPitches = list(set([pitch for pitch in scales.getPitches()]))
        allNoteNames = [i.name for i in allPitches]
        sNoteName = random.choice(allNoteNames)
        lastChordSort = lastChord.sortAscending()
        sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
        sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
        return sNote

    def __generate_approach_tone(self, lastChord):
        """
        Helper function to generate an approach tone.
        """
        sNote = self.__generate_scale_tone(lastChord)
        aNote = sNote.transpose(random.choice([1, -1]))
        return aNote

    def __generate_arbitrary_tone(self, lastChord):
        """
        Helper function to generate a random tone.
        """
        return self.__generate_scale_tone(lastChord) # TODO: make random note

    def parse_melody(self, fullMeasureNotes, fullMeasureChords):
        """
        Given the notes in a measure ('measure') and the chords in that measure
        ('chords'), generate a list of abstract grammatical symbols to represent 
        that measure as described in GTK's "Learning Jazz Grammars" (2009). 

        Inputs: 
        1) "measure" : a stream.Voice object where each element is a
            note.Note or note.Rest object.

            >>> m1
            <music21.stream.Voice 328482572>
            >>> m1[0]
            <music21.note.Rest rest>
            >>> m1[1]
            <music21.note.Note C>

            Can have instruments and other elements, removes them here.

        2) "chords" : a stream.Voice object where each element is a chord.Chord.

            >>> c1
            <music21.stream.Voice 328497548>
            >>> c1[0]
            <music21.chord.Chord E-4 G4 C4 B-3 G#2>
            >>> c1[1]
            <music21.chord.Chord B-3 F4 D4 A3>

            Can have instruments and other elements, removes them here. 

        Outputs:
        1) "fullGrammar" : a string that holds the abstract grammar for measure.
            Format: 
            (Remember, these are DURATIONS not offsets!)
            "R,0.125" : a rest element of  (1/32) length, or 1/8 quarter note. 
            "C,0.125<M-2,m-6>" : chord note of (1/32) length, generated
                                 anywhere from minor 6th down to major 2nd down.
                                 (interval <a,b> is not ordered).
        """
        measure = copy.deepcopy(fullMeasureNotes)
        chords = copy.deepcopy(fullMeasureChords)
        measure.removeByNotOfClass([note.Note, note.Rest])
        chords.removeByNotOfClass([chord.Chord])
        measureStartTime = measure[0].offset - (measure[0].offset % 4)
        measureStartOffset  = measure[0].offset - measureStartTime
        fullGrammar = ""
        prevNote = None
        numNonRests = 0
        for ix, nr in enumerate(measure):
            try: 
                lastChord = [n for n in chords if n.offset <= nr.offset][-1]
            except IndexError:
                chords[0].offset = measureStartTime
                lastChord = [n for n in chords if n.offset <= nr.offset][-1]
            elementType = ' '
            # R: First, check if it's a rest. Clearly a rest --> only one possibility.
            if isinstance(nr, note.Rest):
                elementType = 'R'
            # C: Next, check to see if note pitch is in the last chord.
            elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
                elementType = 'C'
            # L: (Complement tone) Skip this for now.
            # S: Check if it's a scale tone.
            elif self.__is_scale_tone(lastChord, nr):
                elementType = 'S'
            # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
            elif __is_approach_tone(self, lastChord, nr):
                elementType = 'A'
            # X: Otherwise, it's an arbitrary tone. Generate random note.
            else:
                elementType = 'X'
            if (ix == (len(measure)-1)):
                diff = measureStartTime + 4.0 - nr.offset
            else:
                diff = measure[ix + 1].offset - nr.offset
            noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff
            intervalInfo = ""
            if isinstance(nr, note.Note):
                numNonRests += 1
                if numNonRests == 1:
                    prevNote = nr
                else:
                    noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                    noteDistUpper = interval.add([noteDist, "m3"])
                    noteDistLower = interval.subtract([noteDist, "m3"])
                    intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                        noteDistLower.directedName)
                    prevNote = nr
            grammarTerm = noteInfo + intervalInfo 
            fullGrammar += (grammarTerm + " ")
        return fullGrammar.rstrip()

    def unparse_grammar(self, m1_grammar, m1_chords):
        """
        Given a grammar string and chords for a measure, returns measure notes.
        """
        m1_elements = stream.Voice()
        currOffset = 0.0
        prevElement = None
        for ix, grammarElement in enumerate(m1_grammar.split(' ')):
            terms = grammarElement.split(',')
            currOffset += float(terms[1])
            if terms[0] == 'R':
                rNote = note.Rest(quarterLength = float(terms[1]))
                m1_elements.insert(currOffset, rNote)
                continue
            try: 
                lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
            except IndexError:
                m1_chords[0].offset = 0.0
                lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
            if (len(terms) == 2):
                insertNote = note.Note()
                if terms[0] == 'C':
                    insertNote = self.__generate_chord_tone(lastChord)
                elif terms[0] == 'S':
                    insertNote = self.__generate_scale_tone(lastChord)
                else:
                    insertNote = self.__generate_approach_tone(lastChord)
                insertNote.quarterLength = float(terms[1])
                if insertNote.octave < 4:
                    insertNote.octave = 4
                m1_elements.insert(currOffset, insertNote)
                prevElement = insertNote
            else:
                interval1 = interval.Interval(terms[2].replace("<",''))
                interval2 = interval.Interval(terms[3].replace(">",''))
                if interval1.cents > interval2.cents:
                    upperInterval, lowerInterval = interval1, interval2
                else:
                    upperInterval, lowerInterval = interval2, interval1
                lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
                highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
                numNotes = int(highPitch.ps - lowPitch.ps + 1)
                if terms[0] == 'C':
                    relevantChordTones = list()
                    for i in xrange(0, numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self.__is_chord_tone(lastChord, currNote):
                            relevantChordTones.append(currNote)
                    if len(relevantChordTones) > 1:
                        insertNote = random.choice([i for i in relevantChordTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantChordTones) == 1:
                        insertNote = relevantChordTones[0]
                    else:
                        insertNote = prevElement.transpose(random.choice([-2,2]))
                    if insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)
                elif terms[0] == 'S':
                    relevantScaleTones = list()
                    for i in range(numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self.__is_scale_tone(lastChord, currNote):
                            relevantScaleTones.append(currNote)
                    if len(relevantScaleTones) > 1:
                        insertNote = random.choice([i for i in relevantScaleTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantScaleTones) == 1:
                        insertNote = relevantScaleTones[0]
                    else:
                        insertNote = prevElement.transpose(random.choice([-2, 2]))
                    if insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)
                else:
                    relevantApproachTones = list()
                    for i in xrange(0, numNotes):
                        currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                        if self.__is_approach_tone(lastChord, currNote):
                            relevantApproachTones.append(currNote)
                    if len(relevantApproachTones) > 1:
                        insertNote = random.choice([i for i in relevantApproachTones
                            if i.nameWithOctave != prevElement.nameWithOctave])
                    elif len(relevantApproachTones) == 1:
                        insertNote = relevantApproachTones[0]
                    else:
                        insertNote = prevElement.transpose(random.choice([-2, 2]))
                    if insertNote.octave < 3:
                        insertNote.octave = 3
                    insertNote.quarterLength = float(terms[1])
                    m1_elements.insert(currOffset, insertNote)
                prevElement = insertNote
        return m1_elements

class LTSM:
    """
    DOCSTRING
    """
    def build_model(self, corpus, val_indices, max_len, N_epochs=128):
        """
        Build a 2-layer LSTM from a training corpus.
        """
        N_values = len(set(corpus))
        step, sentences, next_values = 3, list(), list()
        for i in range(len(corpus) - max_len, step):
            sentences.append(corpus[i: i + max_len])
            next_values.append(corpus[i + max_len])
        print('nb sequences:', len(sentences))
        X = numpy.zeros((len(sentences), max_len, N_values), dtype=numpy.bool)
        y = numpy.zeros((len(sentences), N_values), dtype=numpy.bool)
        for i, sentence in enumerate(sentences):
            for t, val in enumerate(sentence):
                X[i, t, val_indices[val]] = 1
            y[i, val_indices[next_values[i]]] = 1
        model = keras.models.Sequential()
        model.add(keras.layers.recurrent.LSTM(128, return_sequences=True, input_shape=(max_len, N_values)))
        model.add(keras.layers.core.Dropout(0.2))
        model.add(keras.layers.recurrent.LSTM(128, return_sequences=False))
        model.add(keras.layers.core.Dropout(0.2))
        model.add(keras.layers.core.Dense(N_values))
        model.add(keras.layers.core.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(X, y, batch_size=128, nb_epoch=N_epochs)
        return model

class Preprocess:
    """
    DOCSTRING
    """
    def __get_abstract_grammars(self, measures, chords):
        """
        Helper function to get the grammatical data from given musical data.
        """
        abstract_grammars = list()
        for ix in range(1, len(measures)):
            m = stream.Voice()
            for i in measures[ix]:
                m.insert(i.offset, i)
            c = stream.Voice()
            for j in chords[ix]:
                c.insert(j.offset, j)
            parsed = parse_melody(m, c)
            abstract_grammars.append(parsed)
        return abstract_grammars

    def __parse_midi(self, data_fn):
        """
        Helper function to parse a MIDI file into its measures and chords.
        """
        midi_data = converter.parse(data_fn)
        melody_stream = midi_data[5]
        melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
        for j in melody2:
            melody1.insert(j.offset, j)
        melody_voice = melody1
        for i in melody_voice:
            if i.quarterLength == 0.0:
                i.quarterLength = 0.25
        melody_voice.insert(0, instrument.ElectricGuitar())
        melody_voice.insert(0, key.KeySignature(sharps=1, mode='major'))
        partIndices = [0, 1, 6, 7]
        comp_stream = stream.Voice()
        comp_stream.append([j.flat for i, j in enumerate(midi_data) 
            if i in partIndices])
        full_stream = stream.Voice()
        for i in range(len(comp_stream)):
            full_stream.append(comp_stream[i])
        full_stream.append(melody_voice)
        solo_stream = stream.Voice()
        for part in full_stream:
            curr_part = stream.Part()
            curr_part.append(part.getElementsByClass(instrument.Instrument))
            curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
            curr_part.append(part.getElementsByClass(key.KeySignature))
            curr_part.append(part.getElementsByClass(meter.TimeSignature))
            curr_part.append(part.getElementsByOffset(476, 548, includeEndBoundary=True))
            cp = curr_part.flat
            solo_stream.insert(cp)
        melody_stream = solo_stream[-1]
        measures = collections.OrderedDict()
        offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
        measureNum = 0
        for key_x, group in itertools.groupby(offsetTuples, lambda x: x[0]):
            measures[measureNum] = [n[1] for n in group]
            measureNum += 1
        chordStream = solo_stream[0]
        chordStream.removeByClass(note.Rest)
        chordStream.removeByClass(note.Note)
        offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]
        chords = collections.OrderedDict()
        measureNum = 0
        for key_x, group in itertools.groupby(offsetTuples_chords, lambda x: x[0]):
            chords[measureNum] = [n[1] for n in group]
            measureNum += 1
        del chords[len(chords) - 1]
        assert len(chords) == len(measures)
        return measures, chords

    def get_corpus_data(self, abstract_grammars):
        """
        Get corpus data from grammatical data.
        """
        corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
        values = set(corpus)
        val_indices = dict((v, i) for i, v in enumerate(values))
        indices_val = dict((i, v) for i, v in enumerate(values))
        return corpus, values, val_indices, indices_val

    def get_musical_data(self, data_fn):
        """
        Get musical data from a MIDI file.
        """
        measures, chords = self, __parse_midi(data_fn)
        abstract_grammars = self.__get_abstract_grammars(measures, chords)
        return chords, abstract_grammars

class QA:
    """
    DOCSTRING
    """
    def __grouper(self, iterable, n, fillvalue=None):
        """
        Helper function, from recipes, to iterate over list in chunks of n length.
        """
        args = [iter(iterable)] * n
        return itertools.izip_longest(*args, fillvalue=fillvalue)

    def __roundDown(self, num, mult):
        """
        Helper function to down num to the nearest multiple of mult.
        """
        return (float(num) - (float(num) % mult))

    def __roundUp(self, num, mult):
        """
        Helper function to round up num to nearest multiple of mult.
        """
        return self.__roundDown(num, mult) + mult

    def __roundUpDown(self, num, mult, upDown):
        """
        Helper function that, based on if upDown < 0 or upDown >= 0,
        rounds number down or up respectively to nearest multiple of mult.
        """
        if upDown < 0:
            return self.__roundDown(num, mult)
        else:
            return self.__roundUp(num, mult)

    def clean_up_notes(self, curr_notes):
        """
        Perform quality assurance on notes.
        """
        removeIxs = list()
        for ix, m in enumerate(curr_notes):
            if (m.quarterLength == 0.0):
                m.quarterLength = 0.250
            if (ix < (len(curr_notes) - 1)):
                if (m.offset == curr_notes[ix + 1].offset and
                    isinstance(curr_notes[ix + 1], note.Note)):
                    removeIxs.append((ix + 1))
        curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]
        return curr_notes

    def prune_grammar(self, curr_grammar):
        """
        Smooth the measure, ensuring that everything is in standard note lengths 
        (e.g., 0.125, 0.250, 0.333 ... ).
        """
        pruned_grammar = curr_grammar.split(' ')
        for ix, gram in enumerate(pruned_grammar):
            terms = gram.split(',')
            terms[1] = str(self.__roundUpDown(float(
                terms[1]), 0.250,  random.choice([-1, 1])))
            pruned_grammar[ix] = ','.join(terms)
        pruned_grammar = ' '.join(pruned_grammar)
        return pruned_grammar

    def prune_notes(self, curr_notes):
        """
        Remove repeated notes, and notes that are too close together.
        """
        for n1, n2 in self.__grouper(curr_notes, n=2):
            if n2 == None:
                continue
            if isinstance(n1, note.Note) and isinstance(n2, note.Note):
                if n1.nameWithOctave == n2.nameWithOctave:
                    curr_notes.remove(n2)
        return curr_notes

if __name__ == '__main__':
    GenerateMusic.main(sys.argv)
