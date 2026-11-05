(() => {
    'use strict';
    const canvas = document.getElementById('fluid-flow');
    if (!canvas) return;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');

    // Parameters observed in heybenji.ai's homepage fluid component, 2026-11-05.
    // The reference is a velocity/pressure/dye simulation, not animated noise.
    const config = {
        simulationResolution: 128, dyeResolution: 1024,
        densityDecay: 1.2, velocityDecay: .3,
        pressure: .8, pressureIterations: 20, curl: 25, radius: .2 / 100,
        pointerColor: [.18, .12, .55], pointerForce: 8
    };
    const colors = [
        { value: [.13, .83, .93], weight: .5 },
        { value: [.39, .4, .95], weight: .3 },
        { value: [.2, .83, .6], weight: .12 },
        { value: [.93, .28, .6], weight: .08 }
    ];
    const options = { alpha: true, depth: false, stencil: false, antialias: false };
    const gl = canvas.getContext('webgl2', options) || canvas.getContext('webgl', options);
    if (!gl) { canvas.hidden = true; return; }
    const modern = typeof gl.texStorage2D === 'function';
    const halfFloat = modern ? null : gl.getExtension('OES_texture_half_float');
    const floatRendering = modern ? gl.getExtension('EXT_color_buffer_float') : halfFloat;
    const linear = gl.getExtension(modern ? 'OES_texture_float_linear' : 'OES_texture_half_float_linear');
    if (!floatRendering || (!modern && !linear)) { canvas.hidden = true; return; }
    const type = modern ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    const programs = [], buffers = [];
    let targets = [], velocity, dye, pressure, divergence, curl;
    let frame = 0, lastTime = 0, runningTime = 0, nextBurst = 1.8, seeded = false, lost = false;
    const pending = [];

    const vertex = `
        precision highp float;
        attribute vec2 position;
        uniform vec2 texel;
        varying vec2 uv, leftUV, rightUV, topUV, bottomUV;
        void main() {
            uv = position * .5 + .5;
            leftUV = uv - vec2(texel.x, 0.);
            rightUV = uv + vec2(texel.x, 0.);
            topUV = uv + vec2(0., texel.y);
            bottomUV = uv - vec2(0., texel.y);
            gl_Position = vec4(position, 0., 1.);
        }`;
    const shared = `
        precision highp float;
        precision highp sampler2D;
        varying vec2 uv, leftUV, rightUV, topUV, bottomUV;
    `;
    function compile(shaderType, source) {
        const shader = gl.createShader(shaderType);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const error = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(error);
        }
        return shader;
    }
    function program(source) {
        const p = gl.createProgram();
        const vs = compile(gl.VERTEX_SHADER, vertex), fs = compile(gl.FRAGMENT_SHADER, shared + source);
        gl.attachShader(p, vs); gl.attachShader(p, fs);
        gl.bindAttribLocation(p, 0, 'position');
        gl.linkProgram(p);
        gl.deleteShader(vs); gl.deleteShader(fs);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        const uniforms = {};
        for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
            const name = gl.getActiveUniform(p, i).name;
            uniforms[name] = gl.getUniformLocation(p, name);
        }
        programs.push(p);
        return { p, uniforms };
    }
    let passes;
    try {
        passes = {
            splat: program(`uniform sampler2D source; uniform vec2 point; uniform vec3 color;
                uniform float aspect, radius;
                void main() { vec2 p = uv - point; p.x *= aspect;
                    gl_FragColor = vec4(texture2D(source, uv).xyz + exp(-dot(p,p)/radius)*color, 1.); }`),
            curl: program(`uniform sampler2D velocity;
                void main() { float w = texture2D(velocity,rightUV).y - texture2D(velocity,leftUV).y
                    - texture2D(velocity,topUV).x + texture2D(velocity,bottomUV).x;
                    gl_FragColor = vec4(.5*w,0.,0.,1.); }`),
            vorticity: program(`uniform sampler2D velocity, curl; uniform float strength, dt;
                void main() {
                    float c = texture2D(curl,uv).x;
                    vec2 force = .5 * vec2(abs(texture2D(curl,topUV).x)-abs(texture2D(curl,bottomUV).x),
                        abs(texture2D(curl,rightUV).x)-abs(texture2D(curl,leftUV).x));
                    force /= length(force)+.0001; force *= strength*c; force.y *= -1.;
                    gl_FragColor = vec4(clamp(texture2D(velocity,uv).xy+force*dt,-1000.,1000.),0.,1.);
                }`),
            divergence: program(`uniform sampler2D velocity;
                void main() {
                    float l=texture2D(velocity,leftUV).x, r=texture2D(velocity,rightUV).x;
                    float t=texture2D(velocity,topUV).y, b=texture2D(velocity,bottomUV).y;
                    vec2 c=texture2D(velocity,uv).xy;
                    if(leftUV.x<0.) l=-c.x; if(rightUV.x>1.) r=-c.x;
                    if(topUV.y>1.) t=-c.y; if(bottomUV.y<0.) b=-c.y;
                    gl_FragColor=vec4(.5*(r-l+t-b),0.,0.,1.);
                }`),
            fade: program(`uniform sampler2D source; uniform float amount;
                void main() { gl_FragColor = texture2D(source,uv)*amount; }`),
            pressure: program(`uniform sampler2D pressure, divergence;
                void main() { float p = (texture2D(pressure,leftUV).x + texture2D(pressure,rightUV).x
                    + texture2D(pressure,topUV).x + texture2D(pressure,bottomUV).x
                    - texture2D(divergence,uv).x)*.25; gl_FragColor=vec4(p,0.,0.,1.); }`),
            gradient: program(`uniform sampler2D pressure, velocity;
                void main() { vec2 g=vec2(texture2D(pressure,rightUV).x-texture2D(pressure,leftUV).x,
                    texture2D(pressure,topUV).x-texture2D(pressure,bottomUV).x);
                    gl_FragColor=vec4(texture2D(velocity,uv).xy-g,0.,1.); }`),
            advect: program(`uniform sampler2D velocity, source; uniform vec2 texel;
                uniform float dt, decay;
                void main() { vec2 previous = uv-dt*texture2D(velocity,uv).xy*texel;
                    gl_FragColor=texture2D(source,previous)/(1.+decay*dt); }`),
            display: program(`uniform sampler2D source;
                void main() { vec3 c=texture2D(source,uv).rgb;
                    gl_FragColor=vec4(c,max(c.r,max(c.g,c.b))); }`)
        };
    } catch (error) {
        canvas.hidden = true;
        programs.forEach(p => gl.deleteProgram(p));
        console.warn('Fluid background unavailable:', error.message);
        return;
    }

    const quad = gl.createBuffer(); buffers.push(quad);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.clearColor(0, 0, 0, 0);
    function target(width, height, channels, filter) {
        const texture = gl.createTexture(), fbo = gl.createFramebuffer();
        const t = { texture, fbo, width, height }; targets.push(t);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        const internal = modern ? ({1:gl.R16F, 2:gl.RG16F, 4:gl.RGBA16F})[channels] : gl.RGBA;
        const format = modern ? ({1:gl.RED, 2:gl.RG, 4:gl.RGBA})[channels] : gl.RGBA;
        gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Float framebuffer unsupported');
        gl.viewport(0,0,width,height); gl.clear(gl.COLOR_BUFFER_BIT);
        return t;
    }
    function pair(width, height, channels, filter) {
        return { read:target(width,height,channels,filter), write:target(width,height,channels,filter),
            swap() { [this.read,this.write]=[this.write,this.read]; } };
    }
    function releaseTargets() {
        targets.forEach(t => { gl.deleteTexture(t.texture); gl.deleteFramebuffer(t.fbo); }); targets=[];
    }
    function size(resolution) {
        const ratio=canvas.width/canvas.height;
        return ratio>1 ? [Math.round(resolution*ratio),resolution] : [resolution,Math.round(resolution/ratio)];
    }
    function resize() {
        if (canvas.width === innerWidth && canvas.height === innerHeight && dye) return;
        canvas.width=Math.max(1,innerWidth); canvas.height=Math.max(1,innerHeight);
        releaseTargets();
        const [w,h]=size(config.simulationResolution), [dw,dh]=size(config.dyeResolution);
        dye=pair(dw,dh,4,gl.LINEAR); velocity=pair(w,h,2,gl.LINEAR);
        pressure=pair(w,h,1,gl.NEAREST); divergence=target(w,h,1,gl.NEAREST); curl=target(w,h,1,gl.NEAREST);
    }
    function use(pass) { gl.useProgram(pass.p); return pass.uniforms; }
    function bind(uniform, texture, unit) {
        gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D,texture.texture); gl.uniform1i(uniform,unit);
    }
    function blit(destination) {
        gl.bindFramebuffer(gl.FRAMEBUFFER,destination ? destination.fbo : null);
        gl.viewport(0,0,destination ? destination.width : canvas.width,destination ? destination.height : canvas.height);
        gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    }
    function splat(x,y,dx,dy,color) {
        const u=use(passes.splat);
        gl.uniform2f(u.point,x/canvas.width,1-y/canvas.height);
        gl.uniform1f(u.aspect,canvas.width/canvas.height); gl.uniform1f(u.radius,config.radius);
        bind(u.source,velocity.read,0); gl.uniform3f(u.color,dx,dy,0); blit(velocity.write); velocity.swap();
        bind(u.source,dye.read,0); gl.uniform3fv(u.color,color); blit(dye.write); dye.swap();
    }
    function step(dt) {
        let u=use(passes.curl);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height);
        bind(u.velocity,velocity.read,0); blit(curl);
        u=use(passes.vorticity);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height);
        bind(u.velocity,velocity.read,0); bind(u.curl,curl,1);
        gl.uniform1f(u.strength,config.curl); gl.uniform1f(u.dt,dt); blit(velocity.write); velocity.swap();
        u=use(passes.divergence);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height);
        bind(u.velocity,velocity.read,0); blit(divergence);
        u=use(passes.fade); bind(u.source,pressure.read,0); gl.uniform1f(u.amount,config.pressure);
        blit(pressure.write); pressure.swap();
        u=use(passes.pressure);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height); bind(u.divergence,divergence,0);
        for(let i=0;i<config.pressureIterations;i++) {
            bind(u.pressure,pressure.read,1); blit(pressure.write); pressure.swap();
        }
        u=use(passes.gradient);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height);
        bind(u.pressure,pressure.read,0); bind(u.velocity,velocity.read,1); blit(velocity.write); velocity.swap();
        u=use(passes.advect);
        gl.uniform2f(u.texel,1/velocity.read.width,1/velocity.read.height); gl.uniform1f(u.dt,dt);
        bind(u.velocity,velocity.read,0); bind(u.source,velocity.read,0); gl.uniform1f(u.decay,config.velocityDecay);
        blit(velocity.write); velocity.swap();
        bind(u.velocity,velocity.read,0); bind(u.source,dye.read,1); gl.uniform1f(u.decay,config.densityDecay);
        blit(dye.write); dye.swap();
        u=use(passes.display); bind(u.source,dye.read,0); blit(null);
    }
    function randomColor() {
        let n=Math.random();
        for(const c of colors) { n-=c.weight; if(n<=0) return c.value; }
        return colors[0].value;
    }
    function fail(error) {
        cancelAnimationFrame(frame); frame=0; lost=true; canvas.hidden=true;
        releaseTargets();
        console.warn('Fluid background unavailable:',error.message);
    }
    function draw(now) {
        frame=0;
        try {
            resize();
            const elapsed=Math.min((now-lastTime)/1000,.05); lastTime=now; runningTime+=elapsed;
            gl.disable(gl.BLEND);
            if(!seeded && runningTime>=.5) {
                splat(canvas.width/2,canvas.height/2,0,-80,colors[0].value); seeded=true;
            }
            if(runningTime>=nextBurst) {
                nextBurst=runningTime+1.8;
                if(Math.random()>.4) splat(Math.random()*canvas.width,canvas.height-Math.random()*200,
                    (Math.random()-.5)*40,-Math.random()*60-20,randomColor());
            }
            for(const point of pending.splice(0)) splat(...point);
            step(Math.min(elapsed,.016));
            frame=requestAnimationFrame(draw);
        } catch(error) { fail(error); }
    }
    function active() { return !motion.matches && !document.hidden && !lost; }
    function sync() {
        cancelAnimationFrame(frame); frame=0; pending.length=0;
        if(active()) { lastTime=performance.now(); frame=requestAnimationFrame(draw); }
    }
    function enqueue(x,y,dx,dy) {
        if(!active()) return;
        // Bound pending work if a high-frequency input device outruns rendering.
        if(pending.length>=64) pending.shift();
        pending.push([x,y,dx,dy,config.pointerColor]);
    }
    function mouse(event) {
        enqueue(event.clientX,event.clientY,event.movementX*config.pointerForce,-event.movementY*config.pointerForce);
    }
    function touch(event) {
        const point=event.touches[0];
        if(point) enqueue(point.clientX,point.clientY,10,10);
    }
    // Listen on window, since the decorative canvas must not intercept links or scrolling.
    window.addEventListener('mousemove',mouse,{passive:true});
    window.addEventListener('touchmove',touch,{passive:true});
    document.addEventListener('visibilitychange',sync);
    motion.addEventListener('change',sync);
    canvas.addEventListener('webglcontextlost',() => { lost=true; sync(); canvas.hidden=true; });
    window.addEventListener('pagehide',() => {
        cancelAnimationFrame(frame); frame=0;
    });
    window.addEventListener('pageshow',sync);
    sync();
})();
