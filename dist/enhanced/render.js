import {Renderer as VolumeRenderer} from '../render.js';
import * as optics from './optics.js';

export class Renderer extends VolumeRenderer {
    constructor(gl) {
        super(gl,optics);
    }
}
