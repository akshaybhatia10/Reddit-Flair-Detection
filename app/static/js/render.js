function makeList(comments) {
    var listContainer = document.createElement("div");
    listContainer.id = "comm";
    document.getElementsByTagName("ul")[2].appendChild(listContainer);
    var listElement = document.createElement("ul");
    listContainer.appendChild(listElement);
    
    var numberOfListItems = comments.length;
    if (numberOfListItems > 10){
      numberOfListItems = 10;
    }
    for (var i = 0; i < numberOfListItems; ++i) {
        var listItem = document.createElement("li");
        listItem.innerHTML =comments[i];
        listElement.appendChild(listItem);
    }
}

function again(){
	var url = document.getElementById('url');

  document.getElementById('post').innerHTML = 'Getting Results..Please Wait';
	document.getElementById('para').innerHTML = 'Getting Results..Please Wait';

  var request = new XMLHttpRequest();
    request.onreadystatechange = function() { 
    if (request.readyState == 4 && request.status == 200)
        var data = JSON.parse(request.responseText);
        var comments = [];

        var title = data['title'];
        var self_text = data['selftext']
        var author = data['author']
        var flair = data['flair']
        var comments = data['comments']
        var date = data['date']
        var num_comments = data['num_comments']
        var score = data['score']
        var img_src = data['image_src']

    	  document.getElementById('post').innerHTML = title;
        document.getElementById('author').innerHTML = 'By: ' + author;
        document.getElementById('para').innerHTML = self_text;
        document.getElementById('date').innerHTML = 'Posted on: ' + date;
        document.getElementById('num_comm').innerHTML = 'Total comments: '+ num_comments;
        document.getElementById('score').innerHTML = 'Score: ' + score;
        document.getElementById('flair').innerHTML = 'True flair: ' + flair;
        
        
        makeList(comments)
        console.log(img_src.includes('.com'))
        if (img_src.includes('.com')){
            var listContainer = document.createElement("div");
            listContainer.id = "l";
            document.getElementsByTagName("a")[3].appendChild(listContainer);
            var listElement = document.createElement("a");
            var linkText = document.createTextNode("Source Link");
            listElement .appendChild(linkText);
            listElement.title = "Source Link"
            listElement.href = img_src;
            listContainer.appendChild(listElement);
            }
        else{
          var listContainer = document.createElement("div");
          listContainer.id = "i";
          document.getElementsByTagName("a")[3].appendChild(listContainer);
          var listElement = document.createElement("img");
          listElement.height = "500";
          listElement.width = "500";
          listElement .src = 'https://i.redd.it/' + img_src + '.jpg'
          listContainer.appendChild(listElement);
        }
        
    };
    const endpoint = '/render'
    request.open("POST", endpoint, true); 
    request.setRequestHeader('Access-Control-Allow-Origin', '*');
    request.setRequestHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept')
    request.setRequestHeader('Content-Type', 'application/json');
    request.send(url.value);
}