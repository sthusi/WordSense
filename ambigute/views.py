import re
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import serializers

import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from nltk import pos_tag
from sys import stdout
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
""" import os
from expertai.nlapi.cloud.client import ExpertAiClient
client = ExpertAiClient()

os.environ["EAI_USERNAME"] = 'mahlomolamoses@gmail.com'
os.environ["EAI_PASSWORD"] = '9iP.yVZcy8MStch' """

token ="eyJraWQiOiI1RDVOdFM1UHJBajVlSlVOK1RraXVEZE15WWVMMFJQZ3RaUDJGTlhESHpzPSIsImFsZyI6IlJTMjU2In0.eyJjdXN0b206Y291bnRyeSI6IlpBIiwic3ViIjoiNWUwMjQyMTQtOTVmOS00ZGNhLTkyMDYtMWNjZDA4MGU1MGEwIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC5ldS13ZXN0LTEuYW1hem9uYXdzLmNvbVwvZXUtd2VzdC0xX0FVSGdRMDhDQiIsImNvZ25pdG86dXNlcm5hbWUiOiI1ZTAyNDIxNC05NWY5LTRkY2EtOTIwNi0xY2NkMDgwZTUwYTAiLCJhdWQiOiIxZWdzNjNxOTlwM3NlYmVjaHNiNzI5dDgwbyIsImV2ZW50X2lkIjoiNzZkOWZhMTQtNTJhYi00NzJjLTk1NDEtNmI0NzAyY2Y1ZjM3IiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2Njg0NTk0NDcsIm5hbWUiOiJtYWhsb21vbGEiLCJjdXN0b206am9iVGl0bGUiOiJkZXZlbG9wZXIiLCJleHAiOjE2Njg1NDU4NDcsImlhdCI6MTY2ODQ1OTQ0NywiZmFtaWx5X25hbWUiOiJtb3Rob2dvYW5lIiwiZW1haWwiOiJtYWhsb21vbGFtb3Nlc0BnbWFpbC5jb20iLCJjdXN0b206bWFya2V0aW5nQXV0aCI6IjAifQ.TK6pY1hGoGkc_uKgmGQZSf1T7Md1ZDAypqSkVPoiB1fzEoNkKIYpnZ7RrFbAvfvAF3cMWibOhZ21_OtWFqz8aUPVOuF-UUGbZCYxVE8X0psYCKV8BIFSNo4yymRN2Ta4t1K4keWPDaPamDlEa3L8j0gFVxWCh2QkMrzxtFAs5fH_pnTz4vD4eZhfdkXYYeBNDRl8S-TKaOHJqA1eoxb4DXCRYqrg4ob0ijPJLVEBPFtBEfCFekSrzap8muSThLyoow0z21xeo1offSnfpH4onrjWsSJDaEEX-Iv7BWrRQqmDUz4dK56-p4n5ErfkaA6QTM0ZBCCRZvnEASuRtg_tZQ"

ps = PorterStemmer()

text = "break cut run play make light clear draw give hold set fall take head pass call carry charge point catch check turn close get right cover lift line open go beat drive work roll drop place clean lead raise base blow heavy mark return back block strike good rise stock touch down slip snap keep round see sound square stick white crack direct flat follow order pull stand center dip form ground hit pitch post press settle shift short spread twist black, case, come, deal, face, free, hard, pack, pound, release, shoot, start, stop, support, swing, top, well dead, deep, develop, double, dress, field, fire, hot, jump, key, last, move, pick, step, straight, wash advance, burn, cast, change, control, discharge, fly, have, hook, reduce, sign, soft, spot, strain, stretch, tap, throw bar, breakup, dry, dull, flash, level, live, part, pop, rule, separate, strip address, bank, crash, end, figure, find, flip, flush, force, frame, hang, high, kill, land, loose, match, number, piece, position, present, quarter, rack, rest, rough, scale, score, service, shell, shot, solid, split, stamp, still, tender, tie active, bad, big, bound, drag, drift, exchange, extend, fair, feel, fix, home, job, leave, out, puff, range, register, regular, scratch, shock, slack, spike, squeeze, string, takein, train, trim, upset, walk, wild, yield balance, blast, blue, bolt, choke, cold, cross, crown, do, first, fit, float, foul, fret, hack, hand, issue, low, mean, model, offer, pickup, plate, ride, rush, seat, serve, show, smash, stay, stone, stroke, study, sweet, tight, track, waste act, air, answer, attack, band, bear, book, brush, bull, dark, easy, escape, fast, finish, flare, grain, grant, gray, hunt, lock, loop, master, name, pin, push, rail, represent, review, ring, root, screen, seal, second, section, setup, shake, sharp, sweep, time, tone, transfer, true, wind account, approach, away, bag, be, bond, bow, chip, cloud, color, colour, corner, court, design, die, draft, even, firm, flow, foot, freeze, game, green, guard, heave, house, hurt, jack, kick, life, load, look, march, mold, nose, opening, outside, pit, plug, port, project, raw, render, reverse, secret, shade, shaft, side, stall, stuff, subject, switch, takeout, think, tip, trace, up, wave, young ball, bare, beam, bed, better, bill, bite, board, box, brace, broken, c, card, condition, counter, course, credit, cutting, dig, dirty, express, fresh, front, full, gain, grade, grey, hall, joint, link, major, man, mate, measure, medium, meet, mind, mount, natural, note, patch, pole, positive, process, quiet, rank, reach, real, receive, report, rich, sack, seed, sink, small, spare, spin, splash, superior, takeup, taste, test, thick, thin, thrust, tumble, union, use, view, visit, voice, watch, way, wrong action, average, bang, bend, bitter, bob, body, bottom, bowl, bridge, build, burst, butt, camp, chain, collar, comeup, command, commission, contract, convert, cool, correct, count, crop, crush, cry, dash, date, defense, deliver, deposit, dissolve, division, dock, extension, feed, flag, flick, flight, floor, focus, fold, forward, giveup, glass, glow, help, hitch, horn, just, know, long, minor, miss, negative, net, passing, pay, picture, pinch, prime, proof, puddle, queen, read, recall, record, ruffle, ruin, save, say, shadow, silver, skin, smack, smooth, star, style, tack, tease, thing, title, tough, trade, trap, trip, trust, turnout, warm, weak, wear, wing, withdraw accept, ace, aim, begin, bell, bind, bit, bright, bring, cap, capital, capture, care, carrier, centre, channel, chop, circle, claim, click, clinch, clip, club, collapse, comeout, compound, concentrate, connect, contact, core, crab, cup, cycle, decline, defence, demand, exercise, fail, false, fill, flap, flux, gather, gauge, getoff, grass, grind, guide, heat, heel, hood, host, ice, image, intimate, irregular, jackson, jam, jerk, knock, lap, large, late, like, liquid, mass, mat, material, mature, mould, mouth, movement, new, notice, obscure, offset, operation, opposite, pad, pall, pile, plain, plaster, pocket, pool, power, prick, print, pump, question, radical, rag, rat, rear, reference, relief, relieve, reserve, resistance, resolution, retire, retreat, ridge, rig, running, secure, shape, shine, shower, slick, slow, so, soak, sour, source, spoil, spring, standard, state, stiff, stir, suffer, swallow, swell, tail, talk, transport, troll, trouble, upgrade, value, variation, wheel, whip, whistle, wide, wilson, word allow, alternate, apply, around, badly, bat, bay, belt, bid, blaze, blind, boom, border, bounce, brand, bust, canvass, cat, character, clap, clutch, combine, common, compact, company, complete, concord, content, continue, cradle, crank, curve, day, depression, digest, dim, distribute, divine, drink, dump, edge, edward, empty, engage, explode, expose, exposure, extract, factor, far, fine, flood, fox, function, general, gentle, grace, grand, grip, gross, grow, gum, hammer, heart, hole, idle, inactive, indifferent, inside, interest, introduce, james, king, knot, labor, lean, left, lie, little, living, lodge, lose, lost, love, maintain, makeout, mantle, moderate, nail, narrow, pace, paddle, panel, passage, pattern, peg, perch, plant, plunge, poke, pot, practice, program, projection, purge, put, puton, putout, race, radiate, rally, rap, ray, regard, regenerate, representation, romance, runner, scene, school, scrape, screw, share, sheet, single, sit, skim, slice, slide, slug, smell, smith, smoke, softness, space, spat, special, speed, spell, spill, spiral, spur, stage, stake, steady, stem, sting, stream, strong, submit, suit, sure, surface, tag, tongue, tramp, translate, tread, treat, try, under, undercut, vote, water, waver, weight, west, wish, yoke, zero"

text = text.replace(",", "")
ambg_words = text.split()

class AmbiguteView(APIView):
    def post(self, request):
        text=request.data.get('text')
        ambg_sent = sent_tokenize(text)
        print(ambg_sent)
        res =self.call_lesk(ambg_sent)

        return Response({"data": res})

    
    def lesk(self,context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
        max_overlaps = 0; lesk_sense = None
        context_sentence = context_sentence.split()
        for ss in wn.synsets(ambiguous_word):
            # If POS is specified.
            if pos and ss.pos is not pos:
                continue

            lesk_dictionary = []

            # Includes definition.
            lesk_dictionary+= ss.definition().split()
            # Includes lemma_names.
            lesk_dictionary+= ss.lemma_names()
            # Optional: includes lemma_names of hypernyms and hyponyms.
            if hyperhypo == True:
                lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))       
            if stem == True: # Matching exact words causes sparsity, so lets match stems.
                lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
                context_sentence = [ps.stem(i) for i in context_sentence] 

            overlaps = set(lesk_dictionary).intersection(context_sentence)

            if len(overlaps) > max_overlaps:
                lesk_sense = ss
                max_overlaps = len(overlaps)
        return lesk_sense

    def call_lesk(self,ambg_sent):
        n = len(ambg_sent)
        res=[]
        words=[]
       
      
        for t in range(0,n):
            s=0
            for word in word_tokenize(ambg_sent[t]):
            
                if ps.stem(word) in ambg_words:
                    s = s+1
                    #print("Context:",ambg_sent[t])
                    answer = self.lesk(ambg_sent[t],word)
                    #print("Definition :" ,('%s->' %word) , answer.definition())
                    #print(wn.synsets(word))
                    res.append({
                        "Context":ambg_sent[t],
                        "Definition":(('%s->' %word)+" "+answer.definition()),
                        "word":word,
                        "synsets":wn.synsets(word)[0].pos(),
                        "pos":self.getPos(ambg_sent[0],word)
                       
                    })
            if s == 0:
                #print ("Context:", ambg_sent[t])
                #print ("No Ambiguous word found in this sentence.")
                res.append({
                    "Context": ambg_sent[t],
                    "resultMsg": "Ambiguous word found  in this sentence.",
                    "word":word,
                    "synsets":wn.synsets(word)[0].pos(),
                    "pos":self.getPos(ambg_sent[0],word)
                })
        
        return res

    def getPos(self,ambg_sent,word):
       
        text = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", "", ambg_sent))
        print("tag", nltk.pos_tag(text))
        pos =[]
        highlighted = ""
        for i in text:
            if i == word:
                i = '<u>'+i+'</u>'
            
            syn = wn.synsets(i)
           
           
            flag = not np.any(syn)
            print(i,"**",syn,"**",flag)
            if flag:
                pos.append("")
                
                highlighted = highlighted+" "+i
            else:
                id=self.getPosDef(wn.synsets(i)[0].pos())
                pos.append(id)
                highlighted = highlighted + ' <spam id="'+id+'">'+i+'</spam>'

                print("***",wn.synsets(i))
        return {'pos':pos,'text':highlighted}
    

    def getPosDef(self,pos):
        switch={
            'n':'Noun',
            'v':'Verb',
            'a':'Adjective',
            'r':'adverb',
            }
        return switch.get(pos,"Invalid input")


class aiView(APIView):
    
    def post(self, request):
        self, request
        text=request.data.get('text')

        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)

        print(scores)
       

        return Response({"data": scores})



    
