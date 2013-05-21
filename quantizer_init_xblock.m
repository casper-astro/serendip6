function quantizer_init_xblock(n_signals)
    % create common inputs
    [gain, bit_sel] = xInport('gain', 'bit_sel');

    for i = 1:n_signals
        id = int2str(i);

        % create input port
        din = xInport(['din', id]);

        % create intermediate signal
        mout = xSignal;

        % create output port
        dout = xOutport(['dout', id]);

        % add multiplier
        mult = xBlock('Mult', struct('latency', 3), {din, gain}, {mout});

        % add 4 slice blocks and their output signal paths
        for j = 1:4
            offset = (j - 1) * 8;
            sout{j} = xSignal;
            slice = xBlock('Slice', ...
                            struct('nbits', 8, ...
                                   'mode', 'Lower Bit Location + Width', ...
                                   'bit0', offset), ...
                            {mout}, ...
                            {sout{j}});
        end

        % add mux
        mux = xBlock('Mux', struct('inputs', 4), {bit_sel, sout{1}, sout{2}, sout{3}, sout{4}}, {dout});
    end
end
