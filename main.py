from mido import MidiFile, MidiTrack
import pandas as pd

# open midi file
mid = MidiFile('./RHCP_midi/FortuneFaded.mid', clip=True)
new_mid = MidiFile()

drum_track_number = 0
# find track number of drums
for i in range(len(mid.tracks)):
    if mid.tracks[i][0].channel == 9:
        drum_track_number = i

# find BPM in ticks per beat, and divide it to Thirty-Second 32 notes
new_mid.ticks_per_beat = mid.ticks_per_beat
drum_track = mid.tracks[drum_track_number]
ticks_per_beat_in_32_notes = mid.ticks_per_beat/8

# change notes time to stick it to 32 notes
tmp_time = 0
time_with_note = {}
for i, m in enumerate(drum_track):
    tmp_time += drum_track[i].time
    m.time = round(tmp_time/ticks_per_beat_in_32_notes)
    # make velocity of notes same
    if m.type == 'note_on':
        if m.velocity > 0:
            m.velocity = 1

# crating DataFrame for notes sticked to 32s and filter only note_on notes
transcription = pd.DataFrame(m.dict() for m in drum_track)
transcription = transcription[transcription.type == 'note_on']
# modify table to have columns for every note and lines with time (32 notes as they folow the song)
transcription = transcription.pivot_table(index='time', columns='note', values='velocity', fill_value=0)
# because we have 4/4 tempo, we have to add notes to have folowing 32 notes and empty values we fill with zeros
transcription = transcription.reindex(pd.RangeIndex(transcription.index.max()+1)).fillna(0).sort_index()
# retype to int
transcription = transcription.astype(int)
transcription.columns = transcription.columns.astype(int)
print(transcription)

