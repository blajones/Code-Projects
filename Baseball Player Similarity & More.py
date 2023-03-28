 # This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px


def similarity():
    # read csv file
    baseball = pd.read_csv(".\stats.csv")

    # display DataFrame
    df = pd.DataFrame(baseball, columns= ['b_k_percent', 'b_bb_percent',
        'barrel_batted_rate',	'solidcontact_percent',	'flareburner_percent', 'poorlyunder_percent',
                                               'poorlytopped_percent',	'poorlyweak_percent', 'hard_hit_percent'])

    list_of_single_column = baseball['full'].tolist()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors= len(list_of_single_column))
    name_equivalent, player = send()
    name = name_equivalent
    index_player = df.index.tolist().index(name)
    sim_players = indices[index_player].tolist()
    player_distances = distances[index_player].tolist()
    id_player = sim_players.index(index_player)

    print('Similar Players to ' + str(list_of_single_column[df.index[index_player]]) + ':\n')

    sim_players.remove(index_player)
    player_distances.pop(id_player)
    graph_distance = []
    graph_distance.append(0)
    graph_name = []
    graph_name.append(player)
    j = 1
    length = len(list_of_single_column)
    for i in sim_players:

        graph_distance.append((player_distances[j - 1] * 100))
        graph_name.append(list_of_single_column[df.index[i]])
        print(
                str(j) + ': ' + str(list_of_single_column[df.index[i]]) + ',the distance with' + str(player) + ': ' + str(player_distances[j - 1]))
        j = j + 1

    print('\n')

    playdf = baseball[baseball["full"] == player]
    bkper = playdf['b_k_percent'].tolist()
    bbper = playdf['b_bb_percent'].tolist()
    barrate = playdf['barrel_batted_rate'].tolist()
    solper = playdf['solidcontact_percent'].tolist()
    flareper = playdf['flareburner_percent'].tolist()
    poorun = playdf['poorlyunder_percent'].tolist()
    poortop = playdf['poorlytopped_percent'].tolist()
    poorweak =  playdf['poorlyweak_percent'].tolist()
    hhper = playdf['hard_hit_percent'].tolist()

    playdf2 = baseball[baseball["full"] == graph_name[1]]
    bkper2 = playdf2['b_k_percent'].tolist()
    bbper2 = playdf2['b_bb_percent'].tolist()
    barrate2 = playdf2['barrel_batted_rate'].tolist()
    solper2 = playdf2['solidcontact_percent'].tolist()
    flareper2 = playdf2['flareburner_percent'].tolist()
    poorun2 = playdf2['poorlyunder_percent'].tolist()
    poortop2 = playdf2['poorlytopped_percent'].tolist()
    poorweak2 = playdf2['poorlyweak_percent'].tolist()
    hhper2 = playdf2['hard_hit_percent'].tolist()

    playdf3 = baseball[baseball["full"] == graph_name[2]]
    bkper3 = playdf3['b_k_percent'].tolist()
    bbper3 = playdf3['b_bb_percent'].tolist()
    barrate3 = playdf3['barrel_batted_rate'].tolist()
    solper3 = playdf3['solidcontact_percent'].tolist()
    flareper3 = playdf3['flareburner_percent'].tolist()
    poorun3 = playdf3['poorlyunder_percent'].tolist()
    poortop3 = playdf3['poorlytopped_percent'].tolist()
    poorweak3 = playdf3['poorlyweak_percent'].tolist()
    hhper3 = playdf3['hard_hit_percent'].tolist()

    playdf4 = baseball[baseball["full"] == graph_name[3]]
    bkper4 = playdf4['b_k_percent'].tolist()
    bbper4 = playdf4['b_bb_percent'].tolist()
    barrate4 = playdf4['barrel_batted_rate'].tolist()
    solper4 = playdf4['solidcontact_percent'].tolist()
    flareper4 = playdf4['flareburner_percent'].tolist()
    poorun4 = playdf4['poorlyunder_percent'].tolist()
    poortop4 = playdf4['poorlytopped_percent'].tolist()
    poorweak4 = playdf4['poorlyweak_percent'].tolist()
    hhper4 = playdf4['hard_hit_percent'].tolist()

    playdf5 = baseball[baseball["full"] == graph_name[4]]
    bkper5 = playdf5['b_k_percent'].tolist()
    bbper5 = playdf5['b_bb_percent'].tolist()
    barrate5 = playdf5['barrel_batted_rate'].tolist()
    solper5 = playdf5['solidcontact_percent'].tolist()
    flareper5 = playdf5['flareburner_percent'].tolist()
    poorun5 = playdf5['poorlyunder_percent'].tolist()
    poortop5 = playdf5['poorlytopped_percent'].tolist()
    poorweak5 = playdf5['poorlyweak_percent'].tolist()
    hhper5 = playdf5['hard_hit_percent'].tolist()

    playdf6 = baseball[baseball["full"] == graph_name[5]]
    bkper6 = playdf6['b_k_percent'].tolist()
    bbper6 = playdf6['b_bb_percent'].tolist()
    barrate6 = playdf6['barrel_batted_rate'].tolist()
    solper6 = playdf6['solidcontact_percent'].tolist()
    flareper6 = playdf6['flareburner_percent'].tolist()
    poorun6 = playdf6['poorlyunder_percent'].tolist()
    poortop6 = playdf6['poorlytopped_percent'].tolist()
    poorweak6 = playdf6['poorlyweak_percent'].tolist()
    hhper6 = playdf6['hard_hit_percent'].tolist()

    playdf7 = baseball[baseball["full"] == graph_name[length-1]]
    bkper7 = playdf7['b_k_percent'].tolist()
    bbper7 = playdf7['b_bb_percent'].tolist()
    barrate7 = playdf7['barrel_batted_rate'].tolist()
    solper7 = playdf7['solidcontact_percent'].tolist()
    flareper7 = playdf7['flareburner_percent'].tolist()
    poorun7 = playdf7['poorlyunder_percent'].tolist()
    poortop7 = playdf7['poorlytopped_percent'].tolist()
    poorweak7 = playdf7['poorlyweak_percent'].tolist()
    hhper7 = playdf7['hard_hit_percent'].tolist()

    fig = go.Figure(data=
    go.Scatterpolar(
        r=[graph_distance[0],graph_distance[1],graph_distance[2],graph_distance[3],graph_distance[4],graph_distance[5]],
        #theta=[30,60,90,120,150,180,210,240],
        theta = [65, 15, 210, 110, 312.5, 180, 270],
        mode='markers + text',
        name="Markers and Text",
        text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]],
        textposition= ("top center", "bottom center", "bottom center", "bottom center","bottom center","bottom center")
    ))

    fig.update_layout(showlegend=False)
    fig.show()

    fig = go.Figure(go.Barpolar(
        r=[graph_distance[0] *2,graph_distance[1]*2,graph_distance[2]*2,graph_distance[3]*2,graph_distance[4]*2,graph_distance[5]*2],
        theta=[65, 15, 210, 110, 312.5, 180, 270],
        opacity=0.8,
        #text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]]
    ))

    fig.update_layout(
        template=None,
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
            angularaxis=dict(showticklabels=False, ticks='')
        )
    )

    fig.show()
    fig = go.Figure(data=go.Scatterpolar(
        r=[bkper[0], bbper[0], barrate[0], solper[0], flareper[0], poorun[0], poortop[0], poorweak[0], hhper[0]],
        theta=['b_k_percent', 'b_bb_percent',
        'barrel_batted_rate',	'solidcontact_percent',	'flareburner_percent', 'poorlyunder_percent',
                                               'poorlytopped_percent',	'poorlyweak_percent', 'hard_hit_percent'],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )

    fig.show()

    categories = ['b_k_percent', 'b_bb_percent',
        'barrel_batted_rate',	'solidcontact_percent',	'flareburner_percent', 'poorlyunder_percent',
                                               'poorlytopped_percent',	'poorlyweak_percent', 'hard_hit_percent']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[bkper6[0], bbper6[0], barrate6[0], solper6[0], flareper6[0], poorun6[0], poortop6[0], poorweak6[0],
           hhper6[0]],
        theta=categories,
        fill='toself',
        name=graph_name[5]
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper5[0], bbper5[0], barrate5[0], solper5[0], flareper5[0], poorun5[0], poortop5[0], poorweak5[0],
           hhper5[0]],
        theta=categories,
        fill='toself',
        name=graph_name[4]
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper4[0], bbper4[0], barrate4[0], solper4[0], flareper4[0], poorun4[0], poortop4[0], poorweak4[0],
           hhper4[0]],
        theta=categories,
        fill='toself',
        name=graph_name[3]
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper3[0], bbper3[0], barrate3[0], solper3[0], flareper3[0], poorun3[0], poortop3[0], poorweak3[0],
           hhper3[0]],
        theta=categories,
        fill='toself',
        name=graph_name[2]
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper2[0], bbper2[0], barrate2[0], solper2[0], flareper2[0], poorun2[0], poortop2[0], poorweak2[0], hhper2[0]],
        theta=categories,
        fill='toself',
        name=graph_name[1]
    ))
    fig.add_trace(go.Scatterpolar(
    r = [bkper[0], bbper[0], barrate[0], solper[0], flareper[0], poorun[0], poortop[0], poorweak[0], hhper[0]],
    theta = categories,
    fill = 'toself',
    name = player))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50]
            )),
        showlegend= True
    )

    fig.show()


    X = ['b_k_percent', 'b_bb_percent',
        'barrel_batted_rate',	'solidcontact_percent',	'flareburner_percent', 'poorlyunder_percent',
                                               'poorlytopped_percent',	'poorlyweak_percent', 'hard_hit_percent']
    y1 = [bkper[0], bbper[0], barrate[0], solper[0], flareper[0], poorun[0], poortop[0], poorweak[0], hhper[0]]
    y2 = [bkper2[0], bbper2[0], barrate2[0], solper2[0], flareper2[0], poorun2[0], poortop2[0], poorweak2[0], hhper2[0]]
    y3 = [bkper3[0], bbper3[0], barrate3[0], solper3[0], flareper3[0], poorun3[0], poortop3[0], poorweak3[0],
                  hhper3[0]]
    y4 = [bkper7[0], bbper7[0], barrate7[0], solper7[0], flareper7[0], poorun7[0], poortop7[0], poorweak7[0],
                  hhper7[0]]
    fig = go.Figure(data=[
        go.Bar(name=player, x=X, y=y1,
               text=y1,
               textposition='auto'
               ),
        go.Bar(name=graph_name[1], x=X, y=y2,
               text=y2,
               textposition='auto'
               ),
        go.Bar(name=graph_name[2], x=X,
               y= y3,text= y3,
            textposition='auto'),
        go.Bar(name=graph_name[length-1], x=X,
               y= y4, text=y4,
            textposition='auto')
    ])

    fig.update_layout(barmode='group')
    fig.show()

def rodriguez_similarity():
    baseball = pd.read_csv(".\stats.csv")

    # display DataFrame
    pd.DataFrame(baseball, columns=['b_k_percent', 'b_bb_percent',
                                         'barrel_batted_rate', 'solidcontact_percent', 'flareburner_percent',
                                         'poorlyunder_percent',
                                         'poorlytopped_percent', 'poorlyweak_percent', 'hard_hit_percent'])

    playdf = baseball[baseball["full"] == ' Julio Rodriguez']
    bkper = playdf['b_k_percent'].tolist()
    bbper = playdf['b_bb_percent'].tolist()
    barrate = playdf['barrel_batted_rate'].tolist()
    solper = playdf['solidcontact_percent'].tolist()
    flareper = playdf['flareburner_percent'].tolist()
    poorun = playdf['poorlyunder_percent'].tolist()
    poortop = playdf['poorlytopped_percent'].tolist()
    poorweak = playdf['poorlyweak_percent'].tolist()
    hhper = playdf['hard_hit_percent'].tolist()

    playdf2 = baseball[baseball["full"] == ' Mookie Betts']
    bkper2 = playdf2['b_k_percent'].tolist()
    bbper2 = playdf2['b_bb_percent'].tolist()
    barrate2 = playdf2['barrel_batted_rate'].tolist()
    solper2 = playdf2['solidcontact_percent'].tolist()
    flareper2 = playdf2['flareburner_percent'].tolist()
    poorun2 = playdf2['poorlyunder_percent'].tolist()
    poortop2 = playdf2['poorlytopped_percent'].tolist()
    poorweak2 = playdf2['poorlyweak_percent'].tolist()
    hhper2 = playdf2['hard_hit_percent'].tolist()

    playdf3 = baseball[baseball["full"] == ' Fernando Tatis Jr.']
    bkper3 = playdf3['b_k_percent'].tolist()
    bbper3 = playdf3['b_bb_percent'].tolist()
    barrate3 = playdf3['barrel_batted_rate'].tolist()
    solper3 = playdf3['solidcontact_percent'].tolist()
    flareper3 = playdf3['flareburner_percent'].tolist()
    poorun3 = playdf3['poorlyunder_percent'].tolist()
    poortop3 = playdf3['poorlytopped_percent'].tolist()
    poorweak3 = playdf3['poorlyweak_percent'].tolist()
    hhper3 = playdf3['hard_hit_percent'].tolist()

    playdf4 = baseball[baseball["full"] ==  ' Ronald Acuna Jr.']
    bkper4 = playdf4['b_k_percent'].tolist()
    bbper4 = playdf4['b_bb_percent'].tolist()
    barrate4 = playdf4['barrel_batted_rate'].tolist()
    solper4 = playdf4['solidcontact_percent'].tolist()
    flareper4 = playdf4['flareburner_percent'].tolist()
    poorun4 = playdf4['poorlyunder_percent'].tolist()
    poortop4 = playdf4['poorlytopped_percent'].tolist()
    poorweak4 = playdf4['poorlyweak_percent'].tolist()
    hhper4 = playdf4['hard_hit_percent'].tolist()

    playdf5 = baseball[baseball["full"] == ' Javier Baez']
    bkper5 = playdf5['b_k_percent'].tolist()
    bbper5 = playdf5['b_bb_percent'].tolist()
    barrate5 = playdf5['barrel_batted_rate'].tolist()
    solper5 = playdf5['solidcontact_percent'].tolist()
    flareper5 = playdf5['flareburner_percent'].tolist()
    poorun5 = playdf5['poorlyunder_percent'].tolist()
    poortop5 = playdf5['poorlytopped_percent'].tolist()
    poorweak5 = playdf5['poorlyweak_percent'].tolist()
    hhper5 = playdf5['hard_hit_percent'].tolist()

    playdf6 = baseball[baseball["full"] == ' Cody Bellinger']
    bkper6 = playdf6['b_k_percent'].tolist()
    bbper6 = playdf6['b_bb_percent'].tolist()
    barrate6 = playdf6['barrel_batted_rate'].tolist()
    solper6 = playdf6['solidcontact_percent'].tolist()
    flareper6 = playdf6['flareburner_percent'].tolist()
    poorun6 = playdf6['poorlyunder_percent'].tolist()
    poortop6 = playdf6['poorlytopped_percent'].tolist()
    poorweak6 = playdf6['poorlyweak_percent'].tolist()
    hhper6 = playdf6['hard_hit_percent'].tolist()

    categories = ['b_k_percent', 'b_bb_percent',
                  'barrel_batted_rate', 'solidcontact_percent', 'flareburner_percent', 'poorlyunder_percent',
                  'poorlytopped_percent', 'poorlyweak_percent', 'hard_hit_percent']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[bkper6[0], bbper6[0], barrate6[0], solper6[0], flareper6[0], poorun6[0], poortop6[0], poorweak6[0],
           hhper6[0]],
        theta=categories,
        fill='toself',
        name= 'Cody Bellinger'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper5[0], bbper5[0], barrate5[0], solper5[0], flareper5[0], poorun5[0], poortop5[0], poorweak5[0],
           hhper5[0]],
        theta=categories,
        fill='toself',
        name= 'Javier Baez'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper4[0], bbper4[0], barrate4[0], solper4[0], flareper4[0], poorun4[0], poortop4[0], poorweak4[0],
           hhper4[0]],
        theta=categories,
        fill='toself',
        name= 'Ronald Acuna Jr.'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper3[0], bbper3[0], barrate3[0], solper3[0], flareper3[0], poorun3[0], poortop3[0], poorweak3[0],
           hhper3[0]],
        theta=categories,
        fill='toself',
        name= 'Fernando Tatis Jr.'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper2[0], bbper2[0], barrate2[0], solper2[0], flareper2[0], poorun2[0], poortop2[0], poorweak2[0],
           hhper2[0]],
        theta=categories,
        fill='toself',
        name= 'Mookie Betts'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[bkper[0], bbper[0], barrate[0], solper[0], flareper[0], poorun[0], poortop[0], poorweak[0], hhper[0]],
        theta=categories,
        fill='toself',
        name= 'Julio Rodriguez'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 60]
            )),
        showlegend=True
    )

    fig.show()
def similarity2():
    baseball = pd.read_csv(".\stats_2.csv")

    # display DataFrame
    df = pd.DataFrame(baseball, columns=['b_total_hits', 'b_home_run',
                                         'b_strikeout', 'b_walk', 'batting_avg',
                                         'slg_percent','on_base_percent', 'b_rbi'])

    list_of_single_column = baseball['full'].tolist()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors= len(list_of_single_column))
    name_equivalent, player = send2()
    name = name_equivalent
    # for name  in df.index:

    index_player = df.index.tolist().index(name)
    sim_players = indices[index_player].tolist()
    player_distances = distances[index_player].tolist()
    id_player = sim_players.index(index_player)

    print('Similar Players to ' + str(list_of_single_column[df.index[index_player]]) + ':\n')

    sim_players.remove(index_player)
    player_distances.pop(id_player)
    graph_distance = []
    graph_distance.append(0)
    graph_name = []
    graph_name.append(player)
    j = 1

    for i in sim_players:
        graph_distance.append((player_distances[j - 1] * 100))
        graph_name.append(list_of_single_column[df.index[i]])
        print(
            str(j) + ': ' + str(list_of_single_column[df.index[i]]) + ',the distance with' + str(player) + ': ' + str(
                player_distances[j - 1]))
        j = j + 1
    print('\n')

    fig = go.Figure(data=
    go.Scatterpolar(
        r=[graph_distance[0], graph_distance[1], graph_distance[2], graph_distance[3], graph_distance[4],
           graph_distance[5]],
        # theta=[30,60,90,120,150,180,210,240],
        theta=[65, 15, 210, 110, 312.5, 180, 270],
        mode='markers + text',
        name="Markers and Text",
        text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]],
        textposition=("top center", "bottom right", "bottom center", "bottom center", "bottom center", "bottom center")
    ))

    fig.update_layout(showlegend=False)
    fig.show()

    fig = go.Figure(go.Barpolar(
        r=[graph_distance[0] * 2, graph_distance[1] * 2, graph_distance[2] * 2, graph_distance[3] * 2,
           graph_distance[4] * 2, graph_distance[5] * 2],
        theta=[65, 15, 210, 110, 312.5, 180, 270],
        opacity=0.8,
        # text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]]
    ))



    fig.update_layout(
        template=None,
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
            angularaxis=dict(showticklabels=False, ticks='')
        )
    )

    fig.show()



    playdf = baseball[baseball["full"] == player]
    bkper = playdf['b_total_hits'].tolist()
    bbper = playdf['b_home_run'].tolist()
    barrate = playdf['b_strikeout'].tolist()
    solper = playdf['b_walk'].tolist()
    flareper = playdf['batting_avg'].tolist()
    poorun = playdf['slg_percent'].tolist()
    poortop = playdf['on_base_percent'].tolist()
    poorweak = playdf['b_rbi'].tolist()


    playdf2 = baseball[baseball["full"] == graph_name[1]]
    bkper2 = playdf2['b_total_hits'].tolist()
    bbper2 = playdf2['b_home_run'].tolist()
    barrate2 = playdf2['b_strikeout'].tolist()
    solper2 = playdf2['b_walk'].tolist()
    flareper2 = playdf2['batting_avg'].tolist()
    poorun2 = playdf2['slg_percent'].tolist()
    poortop2 = playdf2['on_base_percent'].tolist()
    poorweak2 = playdf2['b_rbi'].tolist()


    playdf3 = baseball[baseball["full"] == graph_name[2]]
    bkper3 = playdf3['b_total_hits'].tolist()
    bbper3 = playdf3['b_home_run'].tolist()
    barrate3 = playdf3['b_strikeout'].tolist()
    solper3 = playdf3['b_walk'].tolist()
    flareper3 = playdf3['batting_avg'].tolist()
    poorun3 = playdf3['slg_percent'].tolist()
    poortop3 = playdf3['on_base_percent'].tolist()
    poorweak3 = playdf3['b_rbi'].tolist()


    length = len(list_of_single_column)
    playdf7 = baseball[baseball["full"] == graph_name[length - 1]]
    bkper7 = playdf7['b_total_hits'].tolist()
    bbper7 = playdf7['b_home_run'].tolist()
    barrate7 = playdf7['b_strikeout'].tolist()
    solper7 = playdf7['b_walk'].tolist()
    flareper7 = playdf7['batting_avg'].tolist()
    poorun7 = playdf7['slg_percent'].tolist()
    poortop7 = playdf7['on_base_percent'].tolist()
    poorweak7 = playdf7['b_rbi'].tolist()


    X = ['b_total_hits', 'b_home_run','b_strikeout', 'b_walk', 'batting_avg',
                                         'slg_percent','on_base_percent', 'b_rbi']
    y1 = [bkper[0], bbper[0], barrate[0], solper[0], flareper[0], poorun[0], poortop[0], poorweak[0]]
    y2 = [bkper2[0], bbper2[0], barrate2[0], solper2[0], flareper2[0], poorun2[0], poortop2[0], poorweak2[0]]
    y3 = [bkper3[0], bbper3[0], barrate3[0], solper3[0], flareper3[0], poorun3[0], poortop3[0], poorweak3[0]]
    y4 = [bkper7[0], bbper7[0], barrate7[0], solper7[0], flareper7[0], poorun7[0], poortop7[0], poorweak7[0]]
    fig = go.Figure(data=[
        go.Bar(name=player, x=X, y=y1,
               text=y1,
               textposition='auto'
               ),
        go.Bar(name=graph_name[1], x=X, y=y2,
               text=y2,
               textposition='auto'
               ),
        go.Bar(name=graph_name[2], x=X,
               y=y3, text=y3,
               textposition='auto'),
        go.Bar(name=graph_name[length - 1], x=X,
               y=y4, text=y4,
               textposition='auto')
    ])

    fig.update_layout(barmode='group')
    fig.show()
def similarity_pitch():
    baseball = pd.read_csv(".\stats_4.csv")

    # display DataFrame
    df = pd.DataFrame(baseball, columns=['n_ff_formatted','ff_avg_speed','ff_avg_break','n_sl_formatted','sl_avg_speed',
                                         'sl_range_speed','n_ch_formatted','ch_avg_speed','ch_avg_break','n_cukc_formatted',
                                         'cu_avg_speed','cu_avg_break','n_sift_formatted','si_avg_speed','si_avg_break',
                                         'n_fc_formatted','fc_avg_speed','fc_avg_break','n_fs_formatted','fs_avg_speed',
                                         'fs_avg_break','n_kn_formatted','kn_avg_speed','kn_avg_break','n_fastball_formatted',
                                         'fastball_avg_speed','fastball_avg_break','n_breaking_formatted',
                                         'breaking_avg_speed','breaking_avg_break'])

    list_of_single_column = baseball['full'].tolist()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=6)
    name_equivalent, player = send3()
    name = name_equivalent
    # for name  in df.index:

    index_player = df.index.tolist().index(name)
    sim_players = indices[index_player].tolist()
    player_distances = distances[index_player].tolist()
    id_player = sim_players.index(index_player)

    print('Similar Players to ' + str(list_of_single_column[df.index[index_player]]) + ':\n')

    sim_players.remove(index_player)
    player_distances.pop(id_player)
    graph_distance = []
    graph_distance.append(0)
    graph_name = []
    graph_name.append(player)
    j = 1

    for i in sim_players:
        graph_distance.append((player_distances[j - 1] * 100))
        graph_name.append(list_of_single_column[df.index[i]])
        print(
            str(j) + ': ' + str(list_of_single_column[df.index[i]]) + ',the distance with' + str(player) + ': ' + str(
                player_distances[j - 1]))
        j = j + 1
    print('\n')

    fig = go.Figure(data=
    go.Scatterpolar(
        r=[graph_distance[0], graph_distance[1], graph_distance[2], graph_distance[3], graph_distance[4],
           graph_distance[5]],
        # theta=[30,60,90,120,150,180,210,240],
        theta=[65, 15, 210, 110, 312.5, 180, 270],
        mode='markers + text',
        name="Markers and Text",
        text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]],
        textposition=("top center", "bottom right", "bottom center", "bottom center", "bottom center", "bottom center")
    ))

    fig.update_layout(showlegend=False)
    fig.show()

    fig = go.Figure(go.Barpolar(
        r=[graph_distance[0] * 2, graph_distance[1] * 2, graph_distance[2] * 2, graph_distance[3] * 2,
           graph_distance[4] * 2, graph_distance[5] * 2],
        theta=[65, 15, 210, 110, 312.5, 180, 270],
        opacity=0.8,
        # text=[graph_name[0], graph_name[1], graph_name[2], graph_name[3], graph_name[4], graph_name[5]]
    ))

    fig.update_layout(
        template=None,
        polar=dict(
            radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
            angularaxis=dict(showticklabels=False, ticks='')
        )
    )

    fig.show()

def send():
    print("Please pick a player to search" )
    print("(Press enter to continue...)")
    input()

    baseball = pd.read_csv(".\stats.csv")
    list_of_single_column = baseball['full'].tolist()
    print(baseball['full'].values)
    print("You will have to copy and paste player name you choose from list without parenthesis")
    print("Player name: ")
    player = ' '
    search = input(player)
    print("\n")
    i = 0
    for sublist in list_of_single_column:
        i += 1
        if sublist == search:
            print("Found it!", sublist)
            break
    locate = (i - 1)
    return locate, search

def send2():
    print("Please pick a player to search" )
    print("(Press enter to continue...)")
    input()

    baseball = pd.read_csv(".\stats_2.csv")
    list_of_single_column = baseball['full'].tolist()
    print(baseball['full'].values)
    print("You will have to copy and paste player name you choose from list without parenthesis")
    print("Player name: ")
    player = ' '
    search = input(player)
    print("\n")

    i = 0
    for sublist in list_of_single_column:
        i += 1
        if sublist == search:
            print("Found it!", sublist)
            break
    locate = (i - 1)
    return locate, search

def send3():
    print("Please pick a player to search" )
    print("(Press enter to continue...)")
    input()

    baseball = pd.read_csv(".\stats_4.csv")
    list_of_single_column = baseball['full'].tolist()
    print(baseball['full'].values)
    print("You will have to copy and paste player name you choose from list without parenthesis")
    print("Player name: ")
    player = ' '
    search = input(player)
    print("\n")

    i = 0
    for sublist in list_of_single_column:
        i += 1
        if sublist == search:
            print("Found it!", sublist)
            break
    locate = (i - 1)
    return locate, search

def team_plot():
    baseball = pd.read_csv(".\espn_bball_team_stats.csv")

    # display DataFrame
    df = pd.DataFrame(baseball, columns= ['Team', 'wins', 'BA', 'OBP'])

    fig = px.scatter(df, x="BA", y="OBP",
                     color="wins",
                     hover_data=['Team'], trendline="ols")
    fig.show()

    fig = px.scatter_3d(df, x='BA', y='OBP', z='wins',
                        color='Team')
    fig.show()



similarity()
#similarity2()
similarity_pitch()
rodriguez_similarity()
team_plot()