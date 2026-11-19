export const presets = {
  wood: {name:'Wood crib', fuel:.75, wind:.04, turbulence:.64, oxygen:.85},
  pool: {name:'Pool fire', fuel:.84, wind:0, turbulence:.55, oxygen:.78},
  jet: {name:'Jet flame', fuel:.88, wind:0, turbulence:.26, oxygen:1},
  crosswind: {name:'Crosswind', fuel:.83, wind:.78, turbulence:.83, oxygen:.88},
  fireball: {name:'Fireball', fuel:.86, wind:.06, turbulence:.64, oxygen:.9, enhanced:true, oneShot:true, compactDistance:5.6,
    camera:{yaw:.32,pitch:.14,distance:4.25,targetHeight:1.06}},
  groundburst: {name:'Ground burst', fuel:.83, wind:.42, turbulence:.72, oxygen:.85, enhanced:true, oneShot:true, compactDistance:5.9,
    camera:{yaw:.19,pitch:.23,distance:4.1,targetHeight:.61}},
  embers: {name:'Ember bed', fuel:.48, wind:.12, turbulence:.32, oxygen:.66, enhanced:true,
    camera:{yaw:.55,pitch:.46,distance:2.35,targetHeight:.32}},
  pulsing: {name:'Pulsing vent', fuel:.84, wind:.07, turbulence:.4, oxygen:1, enhanced:true,
    camera:{yaw:.23,pitch:.17,distance:2.95,targetHeight:.68}}
};

export const presetCodes = Object.freeze({
  wood:0, pool:1, jet:2, crosswind:3, fireball:4, groundburst:5, embers:6, pulsing:7
});
