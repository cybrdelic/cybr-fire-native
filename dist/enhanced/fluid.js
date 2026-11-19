import {Fluid as OriginalFluid} from '../original/fluid.js';

export class Fluid extends OriginalFluid {
  constructor(gl,quality='medium') {
    super(gl,quality,true);
  }
}
