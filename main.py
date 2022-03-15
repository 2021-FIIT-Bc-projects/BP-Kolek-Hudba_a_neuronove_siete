import keras
import copy
from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo, Message
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import matplotlib.pyplot as plt


def open_midi(midi_file):
    # open midi file
    mid = MidiFile(midi_file, clip=True)

    drum_track_number = 0
    # find track number of drums
    for i in range(len(mid.tracks)):
        for j in range(len(mid.tracks[i])):
            if mid.tracks[i][j].is_meta:
                continue
            if mid.tracks[i][j].channel == 9:
                drum_track_number = i
                break
    return mid, drum_track_number


def get_transcription(drum_track, mid):
    # find ticks per beat, and divide it to Thirty-Second 32 notes
    ticks_per_beat_in_32_notes = mid.ticks_per_beat / 8
    print(mid.ticks_per_beat, ticks_per_beat_in_32_notes)
    # change notes time to stick it to 32 notes
    tmp_time = 0
    note_time_types = {}
    for i, message in enumerate(drum_track):
        # find time how it goes through song
        tmp_time += drum_track[i].time
        print(drum_track[i].time, end=' ')
        note_time_types[drum_track[i].time] = 1
        message.time = round(tmp_time / ticks_per_beat_in_32_notes)
        # make velocity of notes same
        if message.type == 'note_on':
            if message.velocity > 0:
                message.velocity = 1

        print(tmp_time, message.time, tmp_time / ticks_per_beat_in_32_notes, message.type)
    print(note_time_types)

    # crating DataFrame for notes sticked to 32s and filter only note_on notes
    transcription = pd.DataFrame(m.dict() for m in drum_track)
    print(transcription)
    transcription = transcription[transcription.type == 'note_on']
    print(transcription)
    # modify table to have columns for every note and lines with time (32 notes as they folow the song)
    transcription = transcription.pivot_table(index='time', columns='note', values='velocity', fill_value=0)
    print(transcription)
    # because we have 4/4 tempo, we have to add notes to have folowing 32 notes and empty values we fill with zeros
    transcription = transcription.reindex(pd.RangeIndex(transcription.index.max() + 1)).fillna(0).sort_index()
    print(transcription)
    # retype to int
    transcription = transcription.astype(int)
    transcription.columns = transcription.columns.astype(int)
    transcription = transcription.reset_index(drop=True)
    return transcription


def create_midi(tempo, transcription, ticks_per_beat, file_name):
    # create new midi file
    new_mid = MidiFile()
    new_mid.ticks_per_beat = ticks_per_beat
    meta_track = MidiTrack()
    new_mid.tracks.append(meta_track)

    # necessary meta track
    meta_track.append(MetaMessage(type='track_name', name='meta_track', time=0))
    meta_track.append(MetaMessage(type='time_signature', numerator=4, denominator=4, clocks_per_click=24,
                                  notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(MetaMessage(type='set_tempo', tempo=bpm2tempo(tempo), time=0))

    drum_track_new = MidiTrack()
    new_mid.tracks.append(drum_track_new)

    # apend notes to drum track
    ticks_per_32note = int(ticks_per_beat/8)
    notes_from_last_message = 0
    for i, note in enumerate(transcription):
        if i == 0:
            for idx, inst in enumerate(note):
                if inst == 0:
                    continue
                drum_track_new.append(Message('note_on', channel=9, note=instruments[idx], velocity=80, time=0))
            continue
        else:
            if sum(note) < 1:
                notes_from_last_message += 1
                continue
            else:
                notes_from_last_message += 1

            same_note_count = 0
            for idx, inst in enumerate(note):
                if inst == 0:
                    pass
                # if there are more notes at the same time played, they must have time 0
                elif same_note_count == 0:
                    drum_track_new.append(Message('note_on', channel=9, note=instruments[idx], velocity=80,
                                                  time=notes_from_last_message * ticks_per_32note))
                    same_note_count += 1
                else:
                    drum_track_new.append(Message('note_on', channel=9, note=instruments[idx], velocity=80, time=0))
                    same_note_count += 1
            notes_from_last_message = 0
    print(new_mid)
    new_mid.save(file_name)


mid, drum_track_number = open_midi('./RHCP_midi/ZephyrSong.mid')
drum_track = copy.deepcopy(mid.tracks[drum_track_number])

transcription = get_transcription(drum_track, mid)
print(transcription.values)
transcription.to_csv("transcripion.csv")
# find all instruments in song
instruments = transcription.columns.tolist()

bpm = 118
create_midi(bpm, transcription.values, mid.ticks_per_beat, "./output/transcription_new_exp.mid")
#
# inputs_list = []
# outputs_list = []
# sequence_len = 32
# raw_notes = transcription.values
# for i in range(len(raw_notes) - sequence_len):
#     input_start = i
#     input_end = i + sequence_len
#     output_start = input_end
#     output_end = output_start + 1
#
#     # for every 32 notes sequence set next note as output
#     inputs_list.append(raw_notes[input_start:input_end])
#     outputs_list.append(raw_notes[output_start:output_end])
#
# outputs_list = list(np.array(outputs_list).reshape(-1, np.array(outputs_list).shape[-1]))
#
# inputs_list = np.array(inputs_list)
# outputs_list = np.array(outputs_list)
#
# output_shape = outputs_list.shape[1]
# dropout = 0.3
#
# # very very very basic LSTM model
# model = Sequential()
# model.add(LSTM(sequence_len, input_shape=(sequence_len, len(instruments)), return_sequences=True, dropout=dropout))
# model.add(LSTM(sequence_len, return_sequences=True, dropout=dropout))
# model.add(LSTM(sequence_len, dropout=dropout))
# model.add(Dense(output_shape, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',keras.metrics.BinaryCrossentropy()])
#
# mc = ModelCheckpoint(filepath='./new_encode_1st_try.h5', monitor='val_loss', verbose=1, save_best_only=True)
#
# history = model.fit(inputs_list, outputs_list, epochs=10, callbacks=mc, validation_split=0.1, verbose=1, shuffle=False)
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['loss', 'val_loss'], loc='upper left')
#
# plt.show()
#
# # predict new notes
# prediction = model.predict(inputs_list, verbose=0)
# # round prediction to 1 or 0
# prediction = np.around(prediction)
# # retype it to int
# prediction = prediction.astype(int)
#
#
# create_midi(120, prediction, mid.ticks_per_beat)