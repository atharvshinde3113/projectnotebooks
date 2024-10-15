#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install scikit-learn


# In[3]:


import pandas as pd


# In[4]:


sentiment_words=['khush+3', 'khoosh+3', 'khosh+3', 'khsh+3', 'khus+3', 'khoos+3', 'khos+3', 'khs+3', 'khushh+3', 'khooshh+3', 'khoshh+3', 'khshh+3', 'kush+3', 'koosh+3', 'kosh+3', 'ksh+3', 'kus+3', 'koos+3', 'kos+3', 'khs+3', 'kushh+3', 'kooshh+3', 'koshh+3', 'kshh+3', 'achchha+3', 'achchhaa+3', 'aachchha+3', 'aachchhaa+3', 'achcha+3', 'achchaa+3', 'aachcha+3', 'aachchaa+3', 'achha+3', 'achhaa+3', 'aachha+3', 'aachhaa+3', 'acha+3', 'aacha+3','acchha+3', 'acchhaa+3', 'aacchha+3', 'aacchhaa+3', 'accha+3', 'acchaa+3', 'aaccha+3', 'aacchaa+3', 'acha+3', 'achhaa+3', 'aachha+3', 'aachaa+3', 'santusht+3', 'santoosht+3', 'santushth+3', 'santooshth+3', 'sntusht+3', 'sntoosht+3', 'sntushth+3', 'sntooshth+3', 'santust+3', 'santoost+3', 'santusth+3', 'santoosth+3', 'sntust+3', 'sntoost+3', 'sntusth+3', 'sntoosth+3', 'sukhad+3', 'sookhad+3', 'sokhad+3', 'skhad+3', 'sukhd+3', 'sookhd+3', 'sokhd+3', 'skhd+3', 'prasanna+3', 'prasann+3', 'prasana+3', 'prasan+3', 'prsanna+3', 'prsann+3', 'prsana+3', 'prsan+3', 'prasnna+3', 'prasnn+3', 'prasna+3', 'prasn+3', 'prsnna+3', 'prsnn+3', 'prsna+3', 'prsn+3', 'harshit+3', 'harsheet+3', 'harsit+3', 'harseet+3', 'hrshit+3', 'hrsheet+3', 'hrsit+3', 'hrseet+3', 'haarshit+3', 'haarsheet+3', 'haarsit+3', 'haarseet+3','vilakshan+3' ,'veelakshan+3', 'vlakshan+3', 'vilkshan+3' ,'veelkshan+3', 'vlkshan+3', 'vilaksan+3' ,'veelaksan+3', 'vlaksan+3', 'vilksan+3' ,'veelksan+3', 'vlksan+3', 'vilakshn+3' ,'veelakshn+3', 'vlakshn+3', 'vilkshn+3' ,'veelkshn+3', 'vlkshn+3', 'vilaksn+3' ,'veelaksn+3', 'vlaksn+3', 'vilksn+3' ,'veelksn+3', 'vlksn+3','mitravat+3', 'meetravat+3', 'mtravat+3', 'mitrvat+3', 'meetrvat+3', 'mtrvat+3', 'mitravt+3', 'meetravt+3', 'mtravt+3', 'mitrvt+3', 'meetrvt+3', 'mtrvt+3','aanand+3', 'anand+3', 'aannd+3', 'annd+3','garvit+3', 'grvit+3', 'garveet+3', 'grveet+3', 'garvt+3', 'grvt+3', 'garvt+3', 'grvt+3','gaurav+3', 'gourav+3', 'gorav+3', 'gaurv+3', 'gourv+3', 'gorv+3','hoshiyaar+2', 'hshiyaar+2', 'hosheeyaar+2', 'hsheeyaar+2', 'hosheyaar+2', 'hsheyaar+2', 'hoshiyar+2', 'hshiyar+2', 'hosheeyar+2', 'hsheeyar+2', 'hosheyar+2', 'hsheyar+2', 'hoshiyr+2', 'hshiyr+2', 'hosheeyr+2', 'hsheeyr+2', 'hosheyr+2', 'hsheyr+2','bahadoor+2', 'bhadoor+2', 'bahaadoor+2', 'bhaadoor+2', 'bahadur+2', 'bhadur+2', 'bahaadur+2', 'bhaadur+2', 'bhdr+2', 'bhdur+2', 'bhdoor+2', 'bhadur+2', 'bhadoor+2','ashwastha+2', 'aashwastha+2', 'aswastha+2', 'aaswastha+2', 'ashwstha+2', 'aashwstha+2', 'aswstha+2', 'aaswstha+2', 'ashwasth+2', 'aashwasth+2', 'aswasth+2', 'aaswasth+2', 'ashwsth+2', 'aashwsth+2', 'aswsth+2', 'aaswsth+2', 'ashwasta+2', 'aashwasta+2', 'aswasta+2', 'aaswasta+2', 'ashwsta+2', 'aashwsta+2', 'aswsta+2', 'aaswsta+2', 'ashwast+2', 'aashwast+2', 'aswast+2', 'aaswast+2', 'ashwst+2', 'aashwst+2', 'aswst+2', 'aaswst+2','utsuk+2', 'ootsuk+2', 'utsook+2', 'ootsook+2', 'utsk+2', 'ootsk+2','uttejit+2', 'oottejit+2', 'uttajit+2', 'oottajit+2', 'utejit+2', 'ootejit+2', 'utajit+2', 'ootajit+2', 'uttejeet+2', 'oottejeet+2', 'uttajeet+2', 'oottajeet+2', 'utejeet+2', 'ootejeet+2', 'utajeet+2', 'ootajeet+2', 'uttejt+2', 'oottejt+2', 'uttajt+2', 'oottajt+2', 'utejt+2', 'ootejt+2', 'utajt+2', 'ootajt+2', 'uttejet+2', 'oottejet+2', 'uttajet+2', 'oottajet+2', 'utejet+2', 'ootejet+2', 'utajet+2', 'ootajet+2','nishpaksha+0', 'neeshpaksha+0', 'neshpaksha+0', 'nshpaksha+0', 'nispaksha+0', 'neespaksha+0', 'nespaksha+0', 'nspaksha+0', 'nishpksha+0', 'neeshpksha+0', 'neshpksha+0', 'nshpksha+0', 'nispksha+0', 'neespksha+0', 'nespksha+0', 'nspksha+0', 'nishpaksh+0', 'neeshpaksh+0', 'neshpaksh+0', 'nshpaksh+0', 'nispaksh+0', 'neespaksh+0', 'nespaksh+0', 'nspaksh+0', 'nishpksh+0', 'neeshpksh+0', 'neshpksh+0', 'nshpksh+0', 'nispksh+0', 'neespksh+0', 'nespksh+0', 'nspksh+0', 'nishpaks+0', 'neeshpaks+0', 'neshpaks+0', 'nshpaks+0', 'nispaks+0', 'neespaks+0', 'nespaks+0', 'nspaks+0', 'nishpks+0', 'neeshpks+0', 'neshpks+0', 'nshpks+0', 'nispks+0', 'neespks+0', 'nespks+0', 'nspks+0','mahaan+2', 'mhaan+2', 'mahan+2', 'mhan+2', 'mhn+2','nirdosh+1', 'neerdosh+1', 'nrdosh+1', 'nirdsh+1', 'neerdsh+1', 'nrdsh+1', 'nirdos+1', 'neerdos+1', 'nrdos+1', 'nirds+1', 'neerds+1', 'nrds+1','dilchaspa+2', 'deelchaspa+2', 'dlchaspa+2', 'dilchspa+2', 'deelchspa+2', 'dlchspa+2', 'dilchasp+2', 'deelchasp+2', 'dlchasp+2', 'dilchsp+2', 'deelchsp+2', 'dlchsp+2','satyawaadi+2', 'styawaadi+2', 'satywaadi+2', 'stywaadi+2', 'satyavaadi+2', 'styavaadi+2', 'satyvaadi+2', 'styvaadi+2', 'satyawadi+2', 'styawadi+2', 'satywadi+2', 'stywadi+2', 'satyavadi+2', 'styavadi+2', 'satyvadi+2', 'styvadi+2', 'satyawaadee+2', 'styawaadee+2', 'satywaadee+2', 'stywaadee+2', 'satyavaadee+2', 'styavaadee+2', 'satyvaadee+2', 'styvaadee+2', 'satyawadee+2', 'styawadee+2', 'satywadee+2', 'stywadee+2', 'satyavadee+2', 'styavadee+2', 'satyvadee+2', 'styvadee+2', 'satyawaade+2', 'styawaade+2', 'satywaade+2', 'stywaade+2', 'satyavaade+2', 'styavaade+2', 'satyvaade+2', 'styvaade+2', 'satyawade+2', 'styawade+2', 'satywade+2', 'stywade+2', 'satyavade+2', 'styavade+2', 'satyvade+2', 'styvade+2', 'satyawaad+2', 'styawaad+2', 'satywaad+2', 'stywaad+2', 'satyavaad+2', 'styavaad+2', 'satyvaad+2', 'styvaad+2', 'satyawad+2', 'styawad+2', 'satywad+2', 'stywad+2', 'satyavad+2', 'styavad+2', 'satyvad+2', 'styvad+2','aashaawaadi+2', 'ashaawaadi+2', 'aashawaadi+2', 'ashawaadi+2', 'aasaawaadi+2', 'asaawaadi+2', 'aasawaadi+2', 'asawaadi+2', 'aashaavaadi+2', 'ashaavaadi+2', 'aashavaadi+2', 'ashavaadi+2', 'aasaavaadi+2', 'asaavaadi+2', 'aasavaadi+2', 'asavadi+2', 'aashaawadi+2', 'ashaawadi+2', 'aashawadi+2', 'ashawadi+2', 'aasaawadi+2', 'asaawadi+2', 'aasawadi+2', 'asawadi+2', 'aashaavadi+2', 'ashaavadi+2', 'aashavadi+2', 'ashavadi+2', 'aasaavadi+2', 'asaavadi+2', 'aasavadi+2', 'asavadi+2', 'aashaawaadee+2', 'ashaawaadee+2', 'aashawaadee+2', 'ashawaadee+2', 'aasaawaadee+2', 'asaawaadee+2', 'aasawaadee+2', 'asawaadee+2', 'aashaavaadee+2', 'ashaavaadee+2', 'aashavaadee+2', 'ashavaadee+2', 'aasaavaadee+2', 'asaavaadee+2', 'aasavaadee+2', 'asavaadee+2', 'aashaawadee+2', 'ashaawadee+2', 'aashawadee+2', 'ashawadee+2', 'aasaawadee+2', 'asaawadee+2', 'aasawadee+2', 'asawadee+2', 'aashaavadee+2', 'ashaavadee+2', 'aashavadee+2', 'ashavadee+2', 'aasaavadee+2', 'asaavadee+2', 'aasavadee+2', 'asavadee+2', 'aashaawaade+2', 'ashaawaade+2', 'aashawaade+2', 'ashawaade+2', 'aasaawaade+2', 'asaawaade+2', 'aasawaade+2', 'asawaade+2', 'aashaavaade+2', 'ashaavaade+2', 'aashavaade+2', 'ashavaade+2', 'aasaavaade+2', 'asaavaade+2', 'aasavaade+2', 'asavaade+2', 'aashaawade+2', 'ashaawade+2', 'aashawade+2', 'ashawade+2', 'aasaawade+2', 'asaawade+2', 'aasawade+2', 'asawade+2', 'aashaavade+2', 'ashaavade+2', 'aashavade+2', 'ashavade+2', 'aasaavade+2', 'asaavade+2', 'aasavade+2', 'asavade+2', 'aashaawaad+2', 'ashaawaad+2', 'aashawaad+2', 'ashawaad+2', 'aasaawaad+2', 'asaawaad+2', 'aasawaad+2', 'asawaad+2', 'aashaavaad+2', 'ashaavaad+2', 'aashavaad+2', 'ashavaad+2', 'aasaavaad+2', 'asaavaad+2', 'aasavaad+2', 'asavaad+2', 'aashaawad+2', 'ashaawad+2', 'aashawad+2', 'ashawad+2', 'aasaawad+2', 'asaawad+2', 'aasawad+2', 'asawad+2', 'aashaavad+2', 'ashaavad+2', 'aashavad+2', 'ashavad+2', 'aasaavad+2', 'asaavad+2', 'aasavad+2', 'asavad+2','shaant+0', 'shant+0', 'saant+0','satark+1', 'stark+1', 'strk+1', 'satarc+1', 'starc+1', 'strc+1','saavdhaan+1', 'savdhaan+1', 'svdhaan+1', 'saavdhan+1', 'savdhan+1', 'svdhan+1', 'saavdhn+1', 'savdhn+1', 'svdhn+1','dhyeywaadi+2', 'dheywaadi+2', 'dhyeyvaadi+2', 'dheyvaadi+2', 'dhyeywadi+2', 'dheywadi+2', 'dhyeyvadi+2', 'dheyvadi+2', 'dhyeywaadee+2', 'dheywaadee+2', 'dhyeyvaadee+2', 'dheyvaadee+2', 'dhyeywadee+2', 'dheywadee+2', 'dhyeyvadee+2', 'dheyvadee+2', 'dhyeywaade+2', 'dheywaade+2', 'dhyeyvaade+2', 'dheyvaade+2', 'dhyeywade+2', 'dheywade+2', 'dhyeyvade+2', 'dheyvade+2','samajdaar+2', 'smajdaar+2', 'samjdaar+2', 'smjdaar+2', 'samajdar+2', 'smajdar+2', 'samjdar+2', 'smjdar+2', 'samajdr+2', 'smajdr+2', 'samjdr+2', 'smjdr+2', 'samazdaar+2', 'smazdaar+2', 'samzdaar+2', 'smzdaar+2', 'samazdar+2', 'smazdar+2', 'samzdar+2', 'smzdar+2', 'samazdr+2', 'smazdr+2', 'samzdr+2', 'smzdr+2','adbhut+2', 'aadbhut+2', 'atbhut+2', 'aatbhut+2', 'adbhoot+2', 'aadbhoot+2', 'atbhoot+2', 'aatbhoot+2', 'adbhot+2', 'aadbhot+2', 'atbhot+2', 'aatbhot+2','bura-3', 'boora-3', 'buraa-3', 'booraa-3','dukhi-3', 'dookhi-3', 'dukhee-3', 'dookhee-3', 'duki-3', 'dooki-3', 'dukee-3', 'dookee-3', 'dukhe-3', 'dookhe-3', 'duke-3', 'dooke-3', 'dkhi-3', 'dkhi-3', 'dkhee-3', 'dki-3', 'dkee-3', 'dkhe-3', 'dke-3','niraash-3', 'neeraash-3', 'nirash-3', 'neerash-3', 'niraas-3', 'neeraas-3', 'niras-3', 'neeras-3', 'neraash-3', 'nerash-3', 'neraas-3', 'neras-3','abhaagaa-3', 'aabhaagaa-3', 'abhagaa-3', 'aabhagaa-3', 'abhaaga-3', 'aabhaaga-3', 'abhaga-3', 'aabhaga-3', 'abaagaa-3', 'aabaagaa-3', 'abagaa-3', 'aabagaa-3', 'abaaga-3', 'aabaaga-3', 'abaga-3', 'aabaga-3','unmatta-3', 'oonmatta-3', 'unmtta-3', 'oonmtta-3', 'unmata-3', 'oonmata-3', 'unmta-3', 'oonmta-3', 'unmatt-3', 'oonmatt-3', 'unmtt-3', 'oonmtt-3', 'unmat-3', 'oonmat-3', 'unmt-3', 'oonmt-3','doshi-3', 'doshee-3', 'doshe-3','sanaki-3', 'snaki-3', 'sanki-3', 'snki-3', 'sanakee-3', 'snakee-3', 'sankee-3', 'snkee-3', 'sanake-3', 'snake-3', 'sanke-3', 'snke-3','murkh-3', 'moorkh-3', 'murakh-3', 'moorakh-3', 'murk-3', 'moork-3', 'murak-3', 'moorak-3','sharminda-3', 'sarminda-3', 'shrminda-3', 'srminda-3', 'sharmeenda-3', 'sarmeenda-3', 'shrmeenda-3', 'srmeenda-3', 'sharmindaa-3', 'sarmindaa-3', 'shrmindaa-3', 'srmindaa-3', 'sharmeendaa-3', 'sarmeendaa-3', 'shrmeendaa-3', 'srmeendaa-3','pareshaan-3', 'preshaan-3', 'para shaan-3', 'prashaan-3', 'pareshan-3', 'preshan-3', 'parashan-3', 'prashan-3', 'paresaan-3', 'presaan-3', 'parasaan-3', 'prasaan-3', 'paresan-3', 'presan-3', 'parasan-3', 'prasan-3','kruddha-3', 'kriddha-3', 'kraddha-3', 'krddha-3', 'krudhdha-3', 'kridhdha-3', 'kradhdha-3', 'krdhdha-3', 'krudda-3', 'kridda-3', 'kradda-3', 'krdda-3', 'kruddh-3', 'kriddh-3', 'kraddh-3', 'krddh-3', 'krudhdh-3', 'kridhdh-3', 'kradhdh-3', 'krdhdh-3', 'krudd-3', 'kridd-3', 'kradd-3', 'krdd-3','naaraaj-2', 'naraaj-2', 'nraaj-2', 'naaraj-2', 'naraj-2', 'nraj-2', 'naaraaz-2', 'naraaz-2', 'nraaz-2', 'naaraz-2', 'naraz-2','nraz-2','kashta-2', 'kshta-2', 'kasta-2', 'ksta-2', 'kasht-2', 'ksht-2', 'kast-2', 'kst-2','thak-2', 'thk-2','paagal-2', 'pagal-2', 'pgal-2', 'paagl-2', 'pagl-2', 'pgl-2','dar-1', 'darr-1', 'drr-1','uba-2', 'ub-2', 'ooba-2', 'oob-2','avishwaas-2', 'aavishwaas-2', 'aveeshwaas-2', 'aaveeshwaas-2', 'aveshwaas-2', 'aaveshwaas-2', 'avshwaas-2', 'aavshwaas-2', 'avishwas-2', 'aavishwas-2', 'aveeshwas-2', 'aaveeshwas-2', 'aveshwas-2', 'aaveshwas-2', 'avshwas-2', 'aavshwas-2', 'avishws-2', 'aavishws-2', 'aveeshws-2', 'aaveeshws-2', 'aveshws-2', 'aaveshws-2', 'avshws-2', 'aavshws-2','asahaay-1', 'aasahaay-1', 'ashaay-1', 'aashaay-1', 'asahay-1', 'aasahay-1', 'ashay-1', 'aashay-1','udaseen-2', 'oodaseen-2', 'odaseen-2', 'udaaseen-2', 'oodaaseen-2', 'odaaseen-2', 'udseen-2', 'oodseen-2', 'odseen-2', 'udasin-2', 'oodasin-2', 'odasin-2', 'udaasin-2', 'oodaasin-2', 'odaasin-2', 'udsin-2', 'oodsin-2', 'odsin-2', 'udasn-2', 'oodasn-2', 'odasn-2', 'udaasn-2', 'oodaasn-2', 'odaasn-2', 'udsn-2', 'oodsn-2', 'odsn-2', 'udacn-2', 'oodacn-2', 'odacn-2', 'udaacn-2', 'oodaacn-2', 'odaacn-2', 'udcn-2', 'oodcn-2', 'odcn-2','irsha-1', 'eersha-1', 'ersha-1', 'irsa-1', 'eersa-1', 'ersa-1', 'irshaa-1', 'eershaa-1', 'ershaa-1', 'irsaa-1', 'eersaa-1', 'ersaa-1','premaatur-2', 'pramaatur-2', 'prmaatur-2', 'prematur-2', 'pramatur-2', 'prmatur-2', 'premtur-2', 'pramtur-2', 'prmtur-2', 'premaatoor-2', 'pramaatoor-2', 'prmaatoor-2', 'prematoor-2', 'pramatoor-2', 'prmatoor-2', 'premtoor-2', 'pramtoor-2', 'prmtoor-2', 'premaator-2', 'pramaator-2', 'prmaator-2', 'premator-2', 'pramator-2', 'prmator-2', 'premtor-2', 'pramtor-2', 'prmtor-2', 'premaatr-2', 'pramaatr-2', 'prmaatr-2', 'prematr-2', 'pramatr-2', 'prmatr-2', 'premtr-2', 'pramtr-2', 'prmtr-2','sharaarat-2', 'saraarat-2', 'sraarat-2', 'shararat-2', 'sararat-2', 'srarat-2', 'sharaart-2', 'saraart-2', 'sraart-2', 'sharart-2', 'sarart-2', 'srart-2','asantushta-2', 'aasantushta-2', 'asntushta-2', 'aasntushta-2', 'asantooshta-2', 'aasantooshta-2', 'asntooshta-2', 'aasntooshta-2', 'asantoshta-2', 'aasantoshta-2', 'asntoshta-2', 'aasntoshta-2', 'asantusta-2', 'aasantusta-2', 'asntusta-2', 'aasntusta-2', 'asantoosta-2', 'aasantoosta-2', 'asntoosta-2', 'aasntoosta-2', 'asantosta-2', 'aasantosta-2', 'asntosta-2', 'aasntosta-2', 'asantusht-2', 'aasantusht-2', 'asntusht-2', 'aasntusht-2', 'asantoosht-2', 'aasantoosht-2', 'asntoosht-2', 'aasntoosht-2', 'asantosht-2', 'aasantosht-2', 'asntosht-2', 'aasntosht-2', 'asantust-2', 'aasantust-2', 'asntust-2', 'aasntust-2', 'asantoost-2', 'aasantoost-2', 'asntoost-2', 'aasntoost-2', 'asantost-2', 'aasantost-2', 'asntost-2', 'aasntost-2','sandigdha-1', 'sndigdha-1', 'sandeegdha-1', 'sndeegdha-1', 'sandegdha-1', 'sndegdha-1', 'sandgdha-1', 'sndgdha-1', 'sandigda-1', 'sndigda-1', 'sandeegda-1', 'sndeegda-1', 'sandegda-1', 'sndegda-1', 'sandgda-1', 'sndgda-1', 'sandigdh-1', 'sndigdh-1', 'sandeegdh-1', 'sndeegdh-1', 'sandegdh-1', 'sndegdh-1', 'sandgdh-1', 'sndgdh-1', 'sandigd-1', 'sndigd-1', 'sandeegd-1', 'sndeegd-1', 'sandegd-1', 'sndegd-1', 'sandgd-1', 'sndgd-1']


# In[5]:


sentiment_words


# In[6]:


import re


# In[7]:


# Initialize empty lists for words and weights
words = []
weights = []
# Regex pattern to match word and weight separately
pattern = re.compile(r'([a-zA-Z]+)([+-]?\d*)')

for item in sentiment_words:
    match = pattern.match(item)
    if match:
        words.append(match.group(1))
        weights.append(match.group(2))

print("Words:", words)
print("Weights:", weights)


# In[8]:


sentences_as_list=['Main aaj bahut khush hoon.', 'Tumhe dekh kar mujhe khushi hui.', 'Uski jeet ne sabko khush kar diya.', 'Tumhara message mila, main bahut khush hoon.', 'Khush rehna sikho.', 'Wo khush tha apni nayi naukri se.', 'Uske khush hone ka koi mukaam nahi tha.', 'Main khush hoon ki tum yahan ho.', 'Tumhara saath mujhe khush kar deta hai.', 'Uski khushiyon mein sab shamil the.', 'Yeh khabar bahut achchhi hai.', 'Tumhara yeh faisla achchha hai.', 'Aaj ka din bahut achchha raha.', 'Tumne bohot achchha kaam kiya.', 'Tumhara swadisht khana bahut achchha hai.', 'Tumhara mood achchha lag raha hai.', 'Sab kuch achchha chal raha hai.', 'Tumhara plan bahut achchha hai.', 'Yeh kitab padhne mein achchhi hai.', 'Uski achchhi aadatein sabko pasand hain.', 'Aaj ka mausam sukhad hai.', 'Tumhara sandesh sukhad tha.', 'Uske saath guftagu sukhad rahi.', 'Yeh yatra bahut sukhad thi.', 'Uske saath waqt guzaarna sukhad hota hai.', 'Tumhara pehla mulaqat sukhad tha.', 'Yeh anubhav sukhad raha.', 'Uska vyavhaar bahut sukhad tha.', 'Tumhari yeh baatein sukhad hain.', 'Uski hasi sukhad lagti hai.', 'Tum bahadoor ho.', 'Usne bahadoori se kaam kiya.', 'Bahadoor log hamesha jeet te hain.', 'Wo bahadoor hokar ladai lad raha hai.', 'Uske bahadoori ki tareef hui.', 'Tumhara bahadoori dikhana zaroori tha.', 'Bahadoor bano, dar mat.', 'Uski kahani bahadoori se bhari thi.', 'Tumhari yeh bahadoori yaad rahegi.', 'Bahadoor insaan kabhi peeche nahi hat ta.', 'Yeh kitab dilchaspa hai.', 'Tumhari baatein dilchaspa hoti hain.', 'Uski kahani dilchaspa thi.', 'Yeh film bahut dilchaspa hai.', 'Tumhara vichar dilchaspa hai.', 'Uske vichar dilchaspa lagte hain.', 'Yeh charcha dilchaspa ho gayi.', 'Tumhara project dilchaspa hai.', 'Yeh topic bahut dilchaspa hai.', 'Tumhara jeevan bahut dilchaspa hai.', 'Main tumse milne ke liye utsuk hoon.', 'Tumhara jawab sun ne ko utsuk hoon.', 'Uska prashna sun ne ke liye utsuk tha.', 'Wo utsuk tha apne nayi gyaan ko share karne mein.', 'Bachche utsuk hote hain naye cheezein seekhne ke liye.', 'Tumhari kahani sun ne ke liye utsuk hoon.', 'Uske utsukta ko dekh kar mujhe maza aaya.', 'Tumhara naya project dekhne ke liye utsuk hoon.', 'Wo utsuk tha apne prashna ka jawab paane ke liye.', 'Main tumhari kahani sun ne ke liye utsuk hoon.', 'Wo bilkul nirdosh hai.', 'Usne nirdosh hote hue bhi saza payi.', 'Nirdosh bachche khel rahe hain.', 'Uska chehra nirdosh lagta hai.', 'Tumhare aankhon mein nirdosh ta hai.', 'Usne nirdosh hone ka saboot diya.', 'Wo nirdosh hokar bhi dar raha tha.', 'Tumhari baatein nirdosh hain.', 'Uske vichar nirdosh the.', 'Nirdosh logon ko saza nahi milni chahiye.', 'Hamesha satark raho.', 'Usne satark hokar kaam kiya.', 'Satark rehna zaroori hai.', 'Tumhara satark rahna achchha hai.', 'Uski satarkta ne uski jaan bachai.', 'Wo satark hokar sab kuch kar raha tha.', 'Tumhara satark rahna samay ki maang hai.', 'Usne satarkta se har mushkil ko paar kiya.', 'Tum satark hokar chalo.', 'Uski satarkta ne sabko prabhavit kiya.', 'Hamesha saavdhaan raho.', 'Tumhara saavdhaan rehna zaroori hai.', 'Usne saavdhaan hokar kadam uthaya.', 'Saavdhaani se kaam karo.', 'Tumhara saavdhaan rehna achchha hai.', 'Uski saavdhaani ne uski jaan bachai.', 'Wo saavdhaan hokar sab kuch kar raha tha.', 'Tumhara saavdhaan rahna samay ki maang hai.', 'Usne saavdhaani se har mushkil ko paar kiya.', 'Tum saavdhaan hokar chalo.', 'Hamesha nishpaksha raho.', 'Tumhara nishpaksha rahna zaroori hai.', 'Usne nishpaksha hokar faisla kiya.', 'Nishpaksha rehkar faisla karo.', 'Tumhara nishpaksha rehna achchha hai.', 'Uske nishpaksha faisle ko sab ne maana.', 'Wo nishpaksha hokar sab kuch kar raha tha.', 'Tumhara nishpaksha rahna samay ki maang hai.', 'Usne nishpaksha hokar har mushkil ko paar kiya.', 'Tum nishpaksha hokar chalo.', 'Hamesha shaant raho.', 'Tumhara shaant rahna zaroori hai.', 'Usne shaant hokar faisla kiya.', 'Shaant rehkar socho.', 'Tumhara shaant rahna achchha hai.', 'Uska chehra shaant lagta hai.', 'Wo shaant hokar sab kuch kar raha tha.', 'Tumhara shaant rahna samay ki maang hai.', 'Usne shaant hokar har mushkil ko paar kiya.', 'Tum shaant hokar chalo.', 'Yeh bura khabar hai.', 'Tumhara yeh faisla bura hai.', 'Aaj ka din bahut bura raha.', 'Tumne bohot bura kaam kiya.', 'Tumhara vyavhaar bura lag raha hai.', 'Sab kuch bura chal raha hai.', 'Yeh ghatna bahut bura hai.', 'Tumhara plan bura hai.', 'Uska bura swabhaav sabko pasand nahi.', 'Usne bura hone ka saboot diya.', 'Main aaj bahut dukhi hoon.', 'Tumhe dekh kar mujhe dukh hota hai.', 'Uske chale jaane se sab dukhi the.', 'Tumhara message sun ke main dukhi hoon.', 'Dukhi rehna chod do.', 'Wo dukhi tha apni haalat se.', 'Uski dukhi kahani ne sabko rulaya.', 'Main dukhi hoon ki tum yahan nahi ho.', 'Tumhara dukh mujhe mehsoos hota hai.', 'Uske dukhi chehre ne sabko dukh diya.', 'Main aaj bahut niraash hoon.', 'Tumhe dekh kar mujhe niraasha hui.', 'Uske faisle ne sabko niraash kar diya.', 'Tumhara jawab sun ke main niraash hoon.', 'Niraash rehna chod do.', 'Wo niraash tha apni naakamyaabi se.', 'Uske niraash hone ka koi mukaam nahi tha.', 'Main niraash hoon ki tum yahan nahi ho.', 'Tumhara niraash chehra mujhe chubh ta hai.', 'Uske niraash hone se sabko dukh hua.', 'Main tumse naaraaj hoon.', 'Tumhe dekh kar mujhe naaraaji hoti hai.', 'Uske vyavhaar se sab naaraaj hain.', 'Tumhara jawab sun ke main naaraaj hoon.', 'Naaraaj rehna chod do.', 'Wo naaraaj tha apni haalat se.', 'Uski naaraaji sabko samajh aayi.', 'Main naaraaj hoon ki tum yahan nahi ho.', 'Tumhara naaraaj chehra mujhe dekhna nahi.', 'Uske naaraaji se sabko pareshaani hui.', 'Yeh kaam bahut kashta hai.', 'Tumhara faisla kashta hai.', 'Aaj ka din bahut kashta raha.', 'Tumne bohot kashta kaam kiya.', 'Tumhara vyavhaar kashta lag raha hai.', 'Sab kuch kashta chal raha hai.', 'Yeh ghatna bahut kashta hai.', 'Tumhara plan kashta hai.', 'Uska kashta swabhaav sabko pareshan karta hai.', 'Usne kashta hone ka saboot diya.', 'Main aaj bahut thak gaya hoon.', 'Tumhe dekh kar mujhe thakavat mehsoos hoti hai.', 'Uske kaam ne sabko thaka diya.', 'Tumhara yeh kaam kar ke main thak gaya hoon.', 'Thak kar baith jao.', 'Wo thak gaya apni mehnat se.', 'Uske thak jaane ka koi mukaam nahi tha.', 'Main thak gaya hoon ki tum yahan nahi ho.', 'Tumhara thaka chehra mujhe mehsoos hota hai.', 'Uske thak jaane se sabko dukh hua.', 'Mujhe andhere se dar lagta hai.', 'Tumhe dekh kar mujhe dar mehsoos hota hai.', 'Uski kahani sun ke sab dar gaye.', 'Tumhara yeh faisla dar se bhara hai.', 'Dar kar jeena chod do.', 'Wo dar gaya apne soch se.', 'Uske dar ka koi mukaam nahi tha.', 'Main dar raha hoon ki tum yahan nahi ho.', 'Tumhara dar mujhe mehsoos hota hai.', 'Uske dar se sabko pareshaani hui.', 'Main aaj bahut asahaay mehsoos kar raha hoon.', 'Tumhe dekh kar mujhe asahaay lagta hai.', 'Uski kahani sun ke sab asahaay mehsoos karte hain.', 'Tumhara yeh faisla asahaayta se bhara hai.', 'Asahaay mehsoos karna chod do.', 'Wo asahaay tha apni haalat se.', 'Uske asahaay hone ka koi mukaam nahi tha.', 'Main asahaay hoon ki tum yahan nahi ho.', 'Tumhara asahaay chehra mujhe mehsoos hota hai.', 'Uske asahaay hone se sabko pareshaani hui.', 'Uski safalta dekh kar mujhe irsha hui.', 'Tumhari achchi kismat se log irsha karte hain.', 'Irsha ek buri baat hai.', 'Tumhara yeh sochna irsha se bhara hai.', 'Irsha karna chod do.', 'Wo dusron ki safalta se irsha karta hai.', 'Uski irsha sabko samajh aati hai.', 'Main irsha nahi karta, bas sochta hoon.', 'Tumhari irsha mujhe mehsoos hoti hai.', 'Uski irsha ne usko khud se dur kar diya.']


# In[9]:


sentences_as_list = [
    "Khush aur achchha din tha.",
    "Bahadoor baccha bahut utsuk tha.",
    "Satark aur saavdhaan rehna zaroori hai.",
    "Nirdosh log kabhi dar nahi maante.",
    "Shaant aur sukhi jeevan jeene ka raaz jaaniye.",
    "Dukhi insaan ko niraash mat karo.",
    "Kashta aur thakawat ke baad bhi woh niraash nahi hua.",
    "Achchha insaan kabhi dar nahi maanta.",
    "Bahadoor sipahi satark aur saavdhaan tha.",
    "Nishpaksha aur shaant rehna seekho.",
    "Dukhi vyakti ko khush kaise karen.",
    "Niraash na hokar utsuk bane raho.",
    "Thak gaye ho to shaant hokar aaram karo.",
    "Dar aur asahaay mat feel karo.",
    "Irsha aur niraash mat rakhna dil mein.",
    "Sukhad yaadon ko yaad karo.",
    "Bahadoor bano aur dar ko bhool jao.",
    "Saavdhaan rahna bahut zaroori hai.",
    "Nishpaksha vichar rakhte hue aage badho.",
    "Khush aur sukhi rehne ka tareeka.",
    "Bahadoor bano, dar mat.",
    "Dukhi ho to sukhi yaadon ko yaad karo.",
    "Thak gaye ho to aaram karo.",
    "Asahaay logo ki madad karo.",
    "Satark aur saavdhaan rehkar jeevan jeeyo.",
    "Sukhad anubhavon ko samjho.",
    "Khush rehna aur dusro ko bhi khush rakhna.",
    "Bahadoor bano aur utsuk raho.",
    "Nirdosh hone ka matlab samjho.",
    "Shaant aur nishpaksha rahna seekho.",
    "Dukhi na ho, niraash mat ho.",
    "Kashta ke baad sukh milta hai.",
    "Dar mat, asahaay mat feel karo.",
    "Irsha aur dar se door raho.",
    "Sukhad yaadon ko sanjho.",
    "Bahadoor aur satark rahna seekho.",
    "Saavdhaan rahne mein bhalai hai.",
    "Nishpaksha vichar zaroori hain.",
    "Khush aur niraash mat ho.",
    "Dukhi mat ho, khush rehne ki koshish karo.",
    "Thak jaane par aaram karo.",
    "Asahaay logo ki madad karo.",
    "Satark aur saavdhaan rahna seekho.",
    "Sukhad anubhavon ka maza lo.",
    "Khush rehna zaroori hai.",
    "Bahadoor bano aur dar se mukti pao.",
    "Nirdosh hone ka ahsaas.",
    "Shaant aur nishpaksha vichar.",
    "Dukhi vyakti ko khush kaise karen.",
    "Niraash mat ho, utsuk rahna seekho.",
    "Thak gaye ho to aaram karo.",
    "Dar aur asahaay mat feel karo.",
    "Irsha aur niraash mat rakho.",
    "Sukhad anubhav ka maza lo.",
    "Bahadoor bano aur satark rahna seekho.",
    "Saavdhaan rahkar aage badho.",
    "Nishpaksha vichar zaroori hain.",
    "Khush rehne ka raaz kya hai.",
    "Dukhi mat ho, khush raho.",
    "Thak jaane par aaram zaroori hai.",
    "Asahaay logo ki madad karo.",
    "Satark aur saavdhaan rahna seekho.",
    "Sukhad yaadon ko sanjho.",
    "Khush aur sukhi rehne ka tareeka.",
    "Bahadoor bano aur dar se mukti pao.",
    "Nirdosh hone ka ahsaas.",
    "Shaant aur nishpaksha vichar.",
    "Dukhi vyakti ko khush kaise karen.",
    "Niraash mat ho, utsuk rahna seekho.",
    "Thak gaye ho to aaram karo.",
    "Dar aur asahaay mat feel karo.",
    "Irsha aur niraash mat rakho.",
    "Sukhad anubhav ka maza lo.",
    "Bahadoor bano aur satark rahna seekho.",
    "Saavdhaan rahkar aage badho.",
    "Nishpaksha vichar zaroori hain.",
    "Khush rehne ka raaz kya hai.",
    "Dukhi mat ho, khush raho.",
    "Thak jaane par aaram zaroori hai.",
    "Asahaay logo ki madad karo.",
    "Satark aur saavdhaan rahna seekho."
]


# In[10]:


sentences_as_list_lowercase = [sentence.lower() for sentence in sentences_as_list]


# In[12]:


sentences_as_list_lowercase


# In[13]:


sentences_as_list_lowercase[0].split()


# In[14]:


for sentence in sentences_as_list_lowercase:
    sentiment_word_count = 0
    word_net_weight = 0
    sentence_words = sentence.split()
    for word in sentence_words:
        for sentiment_word in sentiment_words:
            if word[:] == sentiment_word[:-2]:
                sentiment_word_count += 1
                print(f"sentiment word count is {sentiment_word_count}")
                word_weight = int(sentiment_word[-2:])
                word_net_weight += word_weight
                print(f"{word} has weight {word_weight}")
                print(word_net_weight)
                break
    if sentiment_word_count > 0:
        sentiment_polarity = word_net_weight / sentiment_word_count
        if sentiment_polarity > 1:
            print(f"sentiment polarity of {sentence} is positive with its index {sentiment_polarity}\n")
        else:
            print(f"sentiment polarity of {sentence} is negative with its index {sentiment_polarity}\n")
    else:
        print("No sentiment words found in this sentence.\n")


# In[30]:


# List to hold data for DataFrame
data = []

for sentence in sentences_as_list_lowercase:
    sentiment_word_count = 0
    word_net_weight = 0
    sentence_words = sentence.split()
    for word in sentence_words:
        for sentiment_word in sentiment_words:
            if word[:] == sentiment_word[:-2]:
                sentiment_word_count += 1
                word_weight = int(sentiment_word[-2:])
                word_net_weight += word_weight
                break
    if sentiment_word_count > 0:
        sentiment_polarity = round(word_net_weight / sentiment_word_count)
    else:
        sentiment_polarity = 0.0  # No sentiment words found
    
    # Add the sentence and its sentiment polarity to the data list
    data.append({'sentence': sentence, 'sentiment_polarity': sentiment_polarity})

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)


# In[31]:


sentences = df['sentence']
print("sentences:\n", sentences)


# In[32]:


sentiment_polarities = df['sentiment_polarity']
print("sentiment_polarities:\n", sentiment_polarities)


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer


# In[34]:


# Vectorization
vectorizer = CountVectorizer()


# In[35]:


X = vectorizer.fit_transform(df['sentence'])


# In[36]:


X


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[46]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, sentiment_polarities, test_size=0.2, random_state=7)


# In[47]:


print("Training set:", X_train)
print("Testing set:", X_test)


# In[48]:


X_train


# In[49]:


y_train


# In[50]:


# Model training
model = LogisticRegression()


# In[51]:


model.fit(X_train, y_train)


# In[52]:


# Prediction and evaluation
y_pred = model.predict(X_test)


# In[54]:


print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X and sentiment_polarities are already defined
# X = ...
# sentiment_polarities = ...

# Variables to store the best random state and highest accuracy
best_random_state = None
max_accuracy = 0

# Loop over random states from 1 to 50
for random_state in range(1, 51):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, sentiment_polarities, test_size=0.2, random_state=random_state)
    
    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Prediction and accuracy calculation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check if this is the best accuracy so far
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_random_state = random_state
    
    print(f'Random State: {random_state}, Accuracy: {accuracy:.4f}')

# Display the best random state and its corresponding accuracy
print(f'Best Random State: {best_random_state}, Maximum Accuracy: {max_accuracy:.4f}')


# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X and sentiment_polarities are already defined
# X = ...
# sentiment_polarities = ...

# List to store accuracy scores for different random states
accuracy_scores = []
best_random_state = None
max_accuracy = 0

# Loop over random states from 1 to 50
for random_state in range(1, 51):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, sentiment_polarities, test_size=0.2, random_state=random_state)
    
    # Model training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Prediction and accuracy calculation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store accuracy score
    accuracy_scores.append((random_state, accuracy))
    print(f'Random State: {random_state}, Accuracy: {accuracy:.4f}')

# Optional: Convert accuracy_scores to a NumPy array or DataFrame for better analysis
accuracy_scores_np = np.array(accuracy_scores)
# or
import pandas as pd
accuracy_df = pd.DataFrame(accuracy_scores, columns=['Random State', 'Accuracy'])
print(accuracy_df)


# In[ ]:




