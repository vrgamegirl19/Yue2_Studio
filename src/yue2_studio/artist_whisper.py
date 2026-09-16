"""Match supplied English lyric words to acoustic Whisper timestamps.

The sequence/fuzzy matching and compact gap interpolation follow the
reference_scene_words timing path used by VRGDG's timestamped lyric extractor.
No ComfyUI code or runtime is imported here.
"""
import difflib
import math


def _norm(text):
    return ''.join(c for c in str(text).casefold() if c.isalnum())


def _acoustic_words(result,duration):
    output=[]
    for segment in getattr(result,'segments',[]) or []:
        for word in getattr(segment,'words',[]) or []:
            text=str(getattr(word,'word','') or '').strip()
            norm=_norm(text)
            start=float(getattr(word,'start',0) or 0)
            end=float(getattr(word,'end',start) or start)
            if norm and math.isfinite(start) and math.isfinite(end):
                output.append(dict(text=text,norm=norm,start=max(0,min(duration,start)),
                                   end=max(0,min(duration,max(start,end)))))
    output.sort(key=lambda x:(x['start'],x['end']))
    return output


def match_lyrics(lyrics,result,duration):
    """Return one ordered timing row per original lyric word and review metadata."""
    from .artist_prepare import words_of
    if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid recording duration for lyric timing.')
    supplied=words_of(lyrics)
    reference=[dict(text=lyrics[start:end],norm=_norm(clean),start_offset=start,
                    end_offset=end,unit_index=lyrics.count('\n',0,start))
               for start,end,clean in supplied]
    acoustic=_acoustic_words(result,duration)
    matched={}
    if acoustic:
        matcher=difflib.SequenceMatcher(None,[w['norm'] for w in reference],
                                        [w['norm'] for w in acoustic],autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):matched[block.a+offset]=(block.b+offset,'exact',1.)
        used={index for index,_,_ in matched.values()}
        for reference_index,ref in enumerate(reference):
            if reference_index in matched:continue
            previous=[a for r,(a,_,_) in matched.items() if r<reference_index]
            following=[a for r,(a,_,_) in matched.items() if r>reference_index]
            lower=max(previous,default=-1)+1;upper=min(following,default=len(acoustic))
            best=None;score=0.
            for index in range(lower,upper):
                if index in used:continue
                ratio=difflib.SequenceMatcher(None,ref['norm'],acoustic[index]['norm']).ratio()
                if ratio>score:best=index;score=ratio
            if best is not None and score>=.68:
                matched[reference_index]=(best,'fuzzy',score)
                used.add(best)
    timed=[None]*len(reference)
    for index,(acoustic_index,kind,match_score) in matched.items():
        seen=acoustic[acoustic_index]
        timed[index]=dict(text=reference[index]['text'],start=seen['start'],end=seen['end'],
                          source=kind,acoustic_text=seen['text'],match_score=round(match_score,3),
                          training_weight=1. if kind=='exact' else .5)
    # Keep unrecognized words near their recognized neighbors, as in the
    # source timing logic. They are review-only; their training weight is zero.
    index=0
    while index<len(timed):
        if timed[index] is not None:index+=1;continue
        first=index
        while index<len(timed) and timed[index] is None:index+=1
        last=index;count=last-first
        before=timed[first-1] if first else None
        after=timed[last] if last<len(timed) else None
        units={reference[i]['unit_index'] for i in range(first,last)}
        before_same=before is not None and reference[first-1]['unit_index'] in units
        after_same=after is not None and reference[last]['unit_index'] in units
        compact=max(.3,count*.35)
        if before and after:
            available_left=before['end'];available_right=max(available_left,after['start'])
            if after_same and not before_same:
                right=available_right;left=max(available_left,right-compact)
            elif before_same and not after_same:
                left=available_left;right=min(available_right,left+compact)
            else:left,right=available_left,available_right
        elif before:left,right=before['end'],min(duration,before['end']+compact)
        elif after:right=after['start'];left=max(0.,right-compact)
        else:left,right=0.,min(duration,compact)
        step=max(.02,(right-left)/count)
        for offset,word_index in enumerate(range(first,last)):
            start=min(duration,left+offset*step)
            end=min(duration,max(start+.02,left+(offset+1)*step))
            timed[word_index]=dict(text=reference[word_index]['text'],start=start,end=end,
                                   source='estimated',acoustic_text=None,match_score=0.,training_weight=0.)
    for ref,word in zip(reference,timed):
        word['start_offset']=ref['start_offset'];word['end_offset']=ref['end_offset']
        word['start']=round(word['start'],3);word['end']=round(word['end'],3)
        word['timing_eligible_for_training']=word['training_weight']>0 and word['end']>word['start']
    counts={kind:sum(word['source']==kind for word in timed) for kind in ('exact','fuzzy','estimated')}
    return timed,dict(duration=round(duration,3),reference_words=len(reference),acoustic_words=len(acoustic),
                      counts=counts,training_words=counts['exact']+counts['fuzzy'])
