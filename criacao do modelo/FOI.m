clear; clc
%salvar nas pastas certas
%tirar if 
%consertar o pop export
%Diretorio com a pasta que está todas as pessoas. 
diretorio_raiz = 'C:\Users\batis\OneDrive\Área de Trabalho\Faculdade\BCI\Dataset\files';
%Diretorio para salvar.
diretoriosave = 'C:\Users\batis\OneDrive\Área de Trabalho\Faculdade\BCI\dados processados2';
%Lista com as epocas. 
lista = ["T0","T1","condition 3"];
%Lista das tarefas que queremos 
tarefas = ["*R03.edf", "*R07.edf", "*R11.edf", "*R04.edf", "*R08.edf", "*R12.edf"]
tarefa = ["R03.edf", "R07.edf", "R11.edf", "R04.edf", "R08.edf", "R12.edf"]
%Pasta das pessoas e quantidade de arquivos nela
pasta_pessoas = dir(diretorio_raiz);
count = 0;
disp("Quantidade de pessoas: "+string(length(pasta_pessoas)));
for e=1:length(tarefas) % for para as tarefas que eu quero 
    for i=1:length(pasta_pessoas) %for para variar as pessoas que eu quero 
        pasta_pessoa_atual= fullfile(diretorio_raiz,pasta_pessoas(i).name);%entrar na pasta da pessoa
        arquivos_edf=dir(fullfile(pasta_pessoa_atual, tarefas{e}));% carrega todos os arquvios edf
        disp(['rodando arquivos da pessoa: ' pasta_pessoa_atual]);
         for j=1:length(arquivos_edf) %for dentro da pessoa para os arquivos
             arquivo_edf_atual= fullfile(pasta_pessoa_atual, arquivos_edf(j).name);
             disp("Rodando arquivo: " + arquivos_edf(j).name);
             filename = char(arquivo_edf_atual);
             disp(string(filename)+" dir");
             for k=1:length(lista)
               EEG = pop_biosig(filename);
               EEG = pop_eegfiltnew(EEG, 'locutoff',8,'hicutoff',13);   
               EEG = pop_epoch( EEG, {  lista(k)  }, [-0.5           4], 'newname', 'EDF file epochs', 'epochinfo', 'yes');
               EEG = pop_rmbase( EEG, [-500 0] ,[]);            
               EEG = pop_eegthresh(EEG,1,[1:64] ,-700,700,-0.5,3.9937,2,0); % components, minimum, maximum, start time, end time
               EEG = pop_rejepoch( EEG, 5,0); %% VER OQ É ESSE TREJ
              % if k == 1 || k == 2 || k == 3          
                    % Salvar arquivos da Tarefa 1
               disp('fullfile: '+string(fullfile(diretoriosave,'teste', string(j)+'.asc')))
               pop_export(EEG, strrep(char(fullfile(diretoriosave,lista(k), tarefa(e)+lista(k)+"p"+string(i)+".asc")),'\','\\'), 'transpose', 'on', 'precision', 4);
                   % count = count + 1;
                    %pop_saveset(EEG, diretoriosave+lista(k));
               disp("Salvo!")
               %elseif k == 4 || k == 5 || k == 6
                    % Salvar arquivos da Tarefa 2 no T0,T1,condition 2
                   % disp('fullfile: '+string(fullfile(diretoriosave,'teste', string(j)+'.asc')))
                   % pop_export(EEG, fullfile(diretoriosave,'teste', string(j)+'.asc'), 'transpose', 'on', 'precision', 4);
                  %  pop_saveset(EEG, diretoriosave+lista(k));
                    %disp("Salvo!")
               
             end
         end
    end
end
disp("Pronto!")
                 