function [Input_sequence,Output_sequence,SamNum,PS]=generate_Smaples(SamNum,tau,step)
    PS=1;
    data = GenerateMakeyGlassData(0.2,0.1,17,1.2+0.2*(rand()-0.5),1,SamNum,1);
    Output_sequence=data(2:size(data,1));
    Input_sequence=data(1:size(data,1)-1);
end
