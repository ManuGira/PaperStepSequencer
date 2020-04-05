import mido
import time

def getJDXiNames():
    while True:
        portnames = mido.get_output_names()
        for pn in portnames:
            if 'JD-Xi' in pn:
                inport = pn[:-1]+'0'
                outport = pn
                return inport, outport
        print("jdxi not found...")
        time.sleep(1)

def getMicrosoftName():
    portnames = mido.get_output_names()
    print(portnames)
    microsoft_outport = [pn for pn in portnames if "microsoft" in pn.lower()]
    if len(microsoft_outport) == 0:
        raise Exception
    return microsoft_outport[0]


class MidiPlayer:
    def __init__(self):
        self.current_note_on = []
        self.outportname = getMicrosoftName()
        self.outport = mido.open_output(self.outportname)
        self.percussion_map = [36, 38, 82, 54]


    def note_on(self, percu_id):
        note = self.percussion_map[percu_id]
        msg = mido.Message('note_on', note=note, channel=9)
        print(msg)
        self.current_note_on.append(msg)
        self.outport.send(msg)

    def note_off_all(self):
        for msg_on in self.current_note_on:
            msg_off = mido.Message('note_off', note=msg_on.note, channel=msg_on.channel)
            self.outport.send(msg_off)
        self.current_note_on = []


if __name__ == '__main__':
    mp = MidiPlayer()
    for percu_id in range(4):
        mp.note_on(percu_id=percu_id)
        time.sleep(0.5)
        mp.note_off_all()
