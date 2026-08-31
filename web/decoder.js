// The wire codec's decode half, lifted VERBATIM from the 14 MB single-file
// build that ran on prod. Not rewritten, not "tidied": the encoder is Python
// in ~/cowork/ascii-cinema/lib/codec.py and these two halves are one contract
// with nothing but tests holding them together. The only edit is the `export`
// on decodeFilm.
//
// Wire format, briefly: films are frames joined by "!", each frame RLE'd
// (`x&12,` = twelve x) and delta-coded against the previous frame in WIRE
// space (`~40_` = copy 40 chars through), with single letters a-h standing in
// for the box-drawing glyphs -- "░" costs 6 bytes as an escape and 1 as "a".

const SEP="\u0021", CA="\u0026", CB="\u002c";
function unrle(s){
  let o="",i=0;
  while(i<s.length){
    const ch=s[i++];
    if(s[i]===CA){const j=s.indexOf(CB,i);o+=ch.repeat(+s.slice(i+1,j));i=j+1;}
    else o+=ch;
  }
  return o;
}
function undelta(prev,d){
  let o="",i=0;
  while(i<d.length){
    if(d[i]==="~"){
      const j=d.indexOf("_",i);const k=+d.slice(i+1,j);
      o+=prev.substr(o.length,k);i=j+1;
    } else o+=d[i++];
  }
  return o;
}
// Wire letters back to display glyphs. Generated from the Python TO_WIRE
// table, never hand-copied, so the two halves cannot drift.
const FROM_WIRE={"a": "\u2591", "b": "\u2592", "c": "\u2593", "d": "\u2588", "e": "\u2500", "f": "\u2502", "g": "\u2571", "h": "\u2572"};
function fromWire(s){
  let o="";
  for(let i=0;i<s.length;i++){const c=s[i];o+=FROM_WIRE[c]||c;}
  return o;
}
export function decodeFilm(wire,cols,rows){
  const parts=wire.split(SEP),out=[];let prev=null;
  for(const p of parts){
    const raw=unrle(p);
    const flat=prev===null?raw:undelta(prev,raw);
    prev=flat;                       // the delta chain stays in WIRE space
    const disp=fromWire(flat);
    let s="";
    for(let r=0;r<rows;r++){ if(r)s+="\n"; s+=disp.substr(r*cols,cols); }
    out.push(s);
  }
  return out;
}
